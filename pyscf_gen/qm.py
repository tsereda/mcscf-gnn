#!/usr/bin/env python3
"""
QM Calculator with PySCF
Usage: python qm.py <input> <methods>
  input:   .inp file or directory of .inp files
  methods: hf,dft,mcscf (comma-separated)

Examples:
  python qm.py molecules/ hf,dft,mcscf
  python qm.py bf.inp hf,mcscf
  python qm.py molecules/ mcscf
"""

import os, time, json, re, sys, subprocess
from pathlib import Path
from pyscf import gto, scf, dft, mcscf
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class InputParser:
    """Parse GAMESS-style .inp files"""
    def __init__(self, inp_file):
        with open(inp_file, 'r') as f:
            self.content = f.read()
        self.inp_file = inp_file
    
    def section(self, name):
        """Extract $SECTION ... $END"""
        m = re.search(rf'\${name}\s*(.*?)\s*\$end', self.content, re.I | re.S)
        return m.group(1).strip() if m else None
    
    def keywords(self, text):
        """Parse key=value pairs"""
        if not text:
            return {}
        kw = {}
        for line in text.split('\n'):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                k, v = k.strip().lower(), v.strip()
                # Parse booleans
                if v.lower() in ['.t.', 'true', '.true.']:
                    v = True
                elif v.lower() in ['.f.', 'false', '.false.']:
                    v = False
                # Parse numbers
                elif v.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
                    v = float(v) if '.' in v or 'e' in v.lower() else int(v)
                kw[k] = v
        return kw
    
    def data(self):
        """Parse $DATA section"""
        text = self.section('data')
        if not text:
            return None, None, []
        
        lines = text.split('\n')
        title = lines[0].strip() if lines else "Unknown"
        
        # Parse symmetry
        sym_line = lines[1].strip().lower() if len(lines) > 1 else ""
        symmetry = None
        if 'c2v' in sym_line or 'cnv' in sym_line:
            symmetry = 'c2v'
        elif 'd6h' in sym_line:
            symmetry = 'd6h'
        elif 'dnh' in sym_line:
            symmetry = 'dnh'
        elif 'td' in sym_line:
            symmetry = 'td'
        
        # Parse atoms
        atoms = []
        start = 2 if symmetry else 1
        for line in lines[start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                # GAMESS format: SYMBOL ATOMIC_NUM X Y Z
                symbol = parts[0]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                atoms.append([symbol, (x, y, z)])
        
        return title, symmetry, atoms
    
    def parse(self):
        """Parse entire .inp file"""
        title, sym, atoms = self.data()
        return {
            'inp_file': str(self.inp_file),
            'title': title,
            'symmetry': sym,
            'atoms': atoms,
            'contrl': self.keywords(self.section('contrl')),
            'system': self.keywords(self.section('system')),
            'basis': self.keywords(self.section('basis')),
            'scf': self.keywords(self.section('scf')),
            'dft': self.keywords(self.section('dft')),
            'mcscf': self.keywords(self.section('mcscf')),
            'det': self.keywords(self.section('det')),
        }


class PySCFRunner:
    """Run calculations with PySCF"""
    
    def __init__(self, methods, output_dir='qm_results'):
        self.methods = [m.strip().lower() for m in methods.split(',')]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def setup_mol(self, inp):
        """Build PySCF molecule"""
        mol = gto.Mole()
        mol.atom = inp['atoms']
        mol.basis = inp['basis'].get('gbasis', 'sto-3g')
        mol.charge = inp['contrl'].get('icharg', 0)
        mol.spin = inp['contrl'].get('mult', 1) - 1
        if inp['symmetry']:
            mol.symmetry = inp['symmetry']
        mol.verbose = 0
        mol.build()
        return mol
    
    def run_hf(self, mol, inp):
        """Run HF"""
        try:
            t0 = time.time()
            scftyp = inp['contrl'].get('scftyp', 'rhf').lower()
            
            if scftyp == 'uhf' or mol.spin > 0:
                mf = scf.UHF(mol)
                name = 'UHF'
            elif scftyp == 'rohf':
                mf = scf.ROHF(mol)
                name = 'ROHF'
            else:
                mf = scf.RHF(mol)
                name = 'RHF'
            
            mf.max_cycle = inp['contrl'].get('maxit', 100)
            mf.conv_tol = inp['scf'].get('conv', 1e-8)
            
            if inp['scf'].get('soscf'):
                mf = mf.newton()
            
            e = mf.kernel()
            return {
                'energy': float(e),
                'time': time.time() - t0,
                'converged': mf.converged,
                'method': name
            }, mf
        except Exception as ex:
            logger.error(f"  HF failed: {ex}")
            return None, None
    
    def run_dft(self, mol, inp):
        """Run DFT"""
        try:
            dft_opts = inp.get('dft', {})
            if not dft_opts.get('dfttyp'):
                logger.warning("  DFT requested but no DFTTYP specified, skipping")
                return None, None
            
            t0 = time.time()
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            
            mf.xc = dft_opts['dfttyp']
            mf.grids.level = dft_opts.get('gridsize', 3)
            mf.max_cycle = inp['contrl'].get('maxit', 100)
            mf.conv_tol = inp['scf'].get('conv', 1e-8)
            
            e = mf.kernel()
            return {
                'energy': float(e),
                'time': time.time() - t0,
                'converged': mf.converged,
                'method': 'DFT',
                'functional': mf.xc
            }, mf
        except Exception as ex:
            logger.error(f"  DFT failed: {ex}")
            return None, None
    
    def run_mcscf(self, mol, inp, mf_guess=None):
        """Run MCSCF"""
        try:
            det = inp.get('det', {})
            if not det:
                logger.warning("  MCSCF requested but no $DET section, skipping")
                return None, None
            
            ncore = det.get('ncore', 0)
            nact = det.get('nact', 4)
            nels = det.get('nels', mol.nelectron)
            
            # Handle spin
            if mol.spin > 0:
                nalpha = (nels + mol.spin) // 2
                nbeta = nels - nalpha
                nelecas = (nalpha, nbeta)
            else:
                nelecas = nels
            
            t0 = time.time()
            
            # Need HF guess
            if mf_guess is None:
                mf_guess = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
                mf_guess.kernel()
            
            cas = mcscf.CASSCF(mf_guess, nact, nelecas)
            cas.max_cycle_macro = inp['mcscf'].get('maxit', 50)
            cas.conv_tol = inp['mcscf'].get('conv', 1e-6)
            
            if not inp['mcscf'].get('fullnr', True):
                cas.ah_start_tol = 1e9
            
            e = cas.kernel()[0]
            return {
                'energy': float(e),
                'time': time.time() - t0,
                'converged': cas.converged,
                'method': 'MCSCF',
                'ncore': ncore,
                'nact': nact,
                'nels': nels
            }, cas
        except Exception as ex:
            logger.error(f"  MCSCF failed: {ex}")
            return None, None
    
    def run(self, inp_file):
        """Run calculation on .inp file"""
        logger.info(f"\n{'='*70}\nPySCF: {inp_file}\n{'='*70}")
        
        # Parse input
        inp = InputParser(inp_file).parse()
        logger.info(f"Title: {inp['title']}")
        
        # Setup molecule
        mol = self.setup_mol(inp)
        logger.info(f"Atoms={mol.natm} Electrons={mol.nelectron} Charge={mol.charge} "
                   f"Mult={inp['contrl'].get('mult', 1)} Basis={mol.basis}")
        
        # Results container
        results = {
            'code': 'pyscf',
            'inp_file': str(inp_file),
            'title': inp['title'],
            'natoms': mol.natm,
            'nelectrons': mol.nelectron,
            'charge': mol.charge,
            'mult': inp['contrl'].get('mult', 1),
            'basis': mol.basis,
            'calculations': {}
        }
        
        # Run requested methods
        mf_guess = None
        
        for method in self.methods:
            if method == 'hf':
                logger.info("  Running HF...")
                r, mf = self.run_hf(mol, inp)
                if r:
                    results['calculations']['HF'] = r
                    mf_guess = mf
            
            elif method == 'dft':
                logger.info("  Running DFT...")
                r, mf = self.run_dft(mol, inp)
                if r:
                    results['calculations']['DFT'] = r
                    if mf_guess is None:
                        mf_guess = mf
            
            elif method == 'mcscf':
                logger.info("  Running MCSCF...")
                det = inp.get('det', {})
                if det:
                    logger.info(f"    NCORE={det.get('ncore')} NACT={det.get('nact')} NELS={det.get('nels')}")
                r, _ = self.run_mcscf(mol, inp, mf_guess)
                if r:
                    results['calculations']['MCSCF'] = r
        
        # Save
        safe_title = inp['title'].replace(' ', '_').replace('/', '_')
        out = self.output_dir / f"{safe_title}.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Saved: {out}")
        
        return results


class GamessRunner:
    """Run calculations with GAMESS"""
    
    def __init__(self, methods, output_dir='qm_results'):
        self.methods = [m.strip().lower() for m in methods.split(',')]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self, inp_file):
        """Run GAMESS calculation"""
        logger.info(f"\n{'='*70}\nGAMESS: {inp_file}\n{'='*70}")
        
        # Parse for metadata
        inp = InputParser(inp_file).parse()
        logger.info(f"Title: {inp['title']}")
        logger.info(f"Methods: {', '.join(self.methods)}")
        
        # Run GAMESS
        logger.info("  Submitting to GAMESS...")
        
        # TODO: Implement actual GAMESS submission
        # This would call rungms or submit to queue
        # For now, just parse the input and return metadata
        
        results = {
            'code': 'gamess',
            'inp_file': str(inp_file),
            'title': inp['title'],
            'methods_requested': self.methods,
            'status': 'not_implemented',
            'note': 'GAMESS submission not yet implemented'
        }
        
        safe_title = inp['title'].replace(' ', '_').replace('/', '_')
        out = self.output_dir / f"{safe_title}_gamess.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"  Note: GAMESS runner not yet implemented")
        logger.info(f"  Metadata saved: {out}")
        
        return results


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("\nError: Wrong number of arguments")
        sys.exit(1)
    
    input_path = sys.argv[1]
    methods = sys.argv[2].lower()
    
    # Validate methods
    valid_methods = {'hf', 'dft', 'mcscf'}
    requested = set(m.strip() for m in methods.split(','))
    if not requested.issubset(valid_methods):
        print(f"Error: methods must be comma-separated from {valid_methods}")
        sys.exit(1)
    
    # Create runner (always PySCF)
    runner = PySCFRunner(methods)
    
    # Collect input files
    path = Path(input_path)
    inp_files = []
    
    if path.is_file() and path.suffix == '.inp':
        inp_files = [path]
    elif path.is_dir():
        inp_files = list(path.glob('*.inp'))
        if not inp_files:
            print(f"Error: No .inp files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: {input_path} is not a .inp file or directory")
        sys.exit(1)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"QM Calculator - PySCF")
    logger.info(f"Methods: {methods.upper()}")
    logger.info(f"Input files: {len(inp_files)}")
    logger.info(f"{'='*70}")
    
    # Run calculations
    all_results = []
    for inp_file in inp_files:
        try:
            result = runner.run(inp_file)
            all_results.append(result)
        except Exception as ex:
            logger.error(f"Failed {inp_file}: {ex}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary = {
        'code': 'pyscf',
        'methods': methods,
        'n_calculations': len(all_results),
        'results': all_results
    }
    
    summary_file = runner.output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE: {len(all_results)} calculations")
    logger.info(f"Summary: {summary_file}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()