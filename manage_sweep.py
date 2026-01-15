#!/usr/bin/env python3
"""
MCSCF-GNN Training Management - Kubernetes Jobs

Usage:
    python manage_training.py                      # Print help
    python manage_training.py --create             # Create sweep + deploy 1 job
    python manage_training.py --create --num 8     # Create sweep + deploy 8 jobs
    python manage_training.py --deploy SWEEP_ID    # Deploy jobs for existing sweep
"""

import os
import sys
import yaml
import wandb
import subprocess
import argparse

# ============================================================================
# WANDB SWEEP MANAGEMENT
# ============================================================================

def load_sweep_config(config_path):
    """Load sweep configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found!")
        sys.exit(1)


def create_sweep(config_path, entity=None, project=None):
    """Create a new wandb sweep and return sweep ID"""
    config = load_sweep_config(config_path)
    
    # Override entity/project if provided
    if entity:
        config['entity'] = entity
    if project:
        config['project'] = project
    
    entity = entity or config.get('entity', 'timgsereda')
    project = project or config.get('project', 'gamess-gnn-sweep')
    
    print(f"\nCreating W&B sweep in {entity}/{project}")
    
    try:
        # Create sweep
        sweep_id = wandb.sweep(config, entity=entity, project=project)
        
        print(f"Sweep created: {sweep_id}")
        print(f"View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"Error creating sweep: {e}")
        return None


# ============================================================================
# KUBERNETES JOB GENERATION
# ============================================================================

def generate_job_yamls(sweep_id, entity, project, output_dir="k8s/training_jobs", num_jobs=4):
    """Generate numbered job YAMLs from template"""
    
    # Read template
    template_path = "tr_job.yml"
    try:
        with open(template_path, 'r') as f:
            template_yaml = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Template file '{template_path}' not found!")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    for i in range(1, num_jobs + 1):
        # Deep copy the template
        job_yaml = yaml.safe_load(yaml.dump(template_yaml))
        
        # Update job name
        job_yaml['metadata']['name'] = f"mcscf-gnn-sweep-{i}"
        
        # Update wandb agent command
        wandb_command = f"wandb agent {entity}/{project}/{sweep_id}"
        
        # Find the args section and update it
        container = job_yaml['spec']['template']['spec']['containers'][0]
        
        # Update the command to include the sweep agent
        container['args'][0] = f"""
              sudo apt-get update && sudo apt-get install -y p7zip-full wget git

              pip install wandb torch_geometric

              git clone https://github.com/tsereda/mcscf-gnn.git
              cd mcscf-gnn

              unzip /data/dec20.zip
              sudo cp -r for-tim/ data/

              {wandb_command}
        """
        
        # Update environment variables
        for env_var in container['env']:
            if env_var['name'] == 'WANDB_PROJECT':
                env_var['value'] = project
            elif env_var['name'] == 'WANDB_ENTITY':
                env_var['value'] = entity
        
        # Write to file
        output_file = os.path.join(output_dir, f"mcscf-gnn-sweep-job-{i}.yml")
        
        with open(output_file, 'w') as f:
            yaml.dump(job_yaml, f, default_flow_style=False, sort_keys=False)
        
        generated_files.append(output_file)
        print(f"Generated: {output_file}")
    
    return generated_files


def deploy_jobs(job_files):
    """Deploy jobs to Kubernetes"""
    
    print(f"\nDeploying {len(job_files)} training jobs to Kubernetes...")
    
    for job_file in job_files:
        job_name = os.path.basename(job_file).replace('.yml', '')
        
        try:
            print(f"\n  Deploying {job_name}...")
            result = subprocess.run(
                ['kubectl', 'apply', '-f', job_file],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  {job_name}: {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as e:
            print(f"  Error: {job_name} failed: {e.stderr}")
        except FileNotFoundError:
            print(f"  Error: kubectl not found. Please install kubectl or deploy manually:")
            print(f"     kubectl apply -f {job_file}")
            return False
    
    return True


def delete_jobs():
    """Delete all wandb sweep jobs from Kubernetes"""
    
    print("\nDeleting all wandb sweep jobs...")
    
    try:
        # First, list the jobs that will be deleted
        list_result = subprocess.run(
            ['kubectl', 'get', 'jobs', '-l', 'app=wandb-sweep', '-o', 'name'],
            capture_output=True,
            text=True,
            check=True
        )
        
        jobs = [j.strip() for j in list_result.stdout.strip().split('\n') if j.strip()]
        
        if not jobs:
            print("  No jobs found with label app=wandb-sweep")
            return True
        
        print(f"  Found {len(jobs)} job(s) to delete:")
        for job in jobs:
            print(f"    - {job}")
        
        # Confirm deletion
        confirm = input("\n  Delete these jobs? [y/N]: ").lower().strip()
        
        if confirm != 'y':
            print("  Deletion cancelled")
            return False
        
        # Delete jobs
        result = subprocess.run(
            ['kubectl', 'delete', 'jobs', '-l', 'app=wandb-sweep'],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"\n  Deleted successfully:")
        print(f"  {result.stdout.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  Error: Failed to delete jobs: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  Error: kubectl not found. Please delete manually:")
        print(f"     kubectl delete jobs -l app=wandb-sweep")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MCSCF-GNN Training Management - Create sweep + deploy training jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sweep + deploy 1 job
  python manage_training.py --create
  
  # Create sweep + deploy 8 jobs
  python manage_training.py --create --num 8
  
  # Deploy jobs to existing sweep
  python manage_training.py --deploy dnffyu6j
  
  # Custom entity/project
  python manage_training.py --entity myteam --project myproject

Tip:
  kubectl get jobs -l app=wandb-sweep
  kubectl logs -f job/mcscf-gnn-sweep-1
  python manage_training.py --delete
        """
    )
    
    parser.add_argument('--create', action='store_true',
                       help='Create sweep and deploy jobs')
    parser.add_argument('--deploy', type=str, metavar='SWEEP_ID',
                       help='Deploy jobs for existing sweep ID')
    parser.add_argument('--delete', action='store_true',
                       help='Delete all wandb sweep jobs')
    parser.add_argument('--num', type=int, default=1,
                       help='Number of jobs to deploy (default: 1)')
    parser.add_argument('--entity', type=str, default='timgsereda',
                       help='W&B entity (default: timgsereda)')
    parser.add_argument('--project', type=str, default='gamess-gnn-sweep',
                       help='W&B project (default: gamess-gnn-sweep)')
    parser.add_argument('sweep_file', nargs='?', default='sweep.yml',
                       help='Path to sweep configuration file (default: sweep.yml)')
    
    args = parser.parse_args()
    
    # If no action specified, print help
    if not args.create and not args.deploy and not args.delete:
        parser.print_help()
        sys.exit(0)
    
    # Handle deletion
    if args.delete:
        print("=" * 70)
        print("MCSCF-GNN Training Management - Delete Jobs")
        print("=" * 70)
        delete_jobs()
        sys.exit(0)
    
    print("=" * 70)
    print("MCSCF-GNN Training Management")
    print("=" * 70)
    
    # Create or use existing sweep
    if args.deploy:
        sweep_id = args.deploy
        print(f"\nUsing existing sweep: {sweep_id}")
    elif args.create:
        print("\n[Step 1/3] Creating W&B sweep...")
        sweep_id = create_sweep(
            config_path=args.sweep_file,
            entity=args.entity, 
            project=args.project
        )
        
        if not sweep_id:
            print("Error: Failed to create sweep. Aborting...")
            sys.exit(1)
    
    # Generate job YAMLs
    print(f"\n[Step 2/3] Generating {args.num} Kubernetes job YAMLs...")
    job_files = generate_job_yamls(
        sweep_id=sweep_id,
        entity=args.entity,
        project=args.project,
        num_jobs=args.num
    )
    
    print(f"\nJob YAMLs saved to: k8s/training_jobs/")
    
    # Deploy jobs
    print(f"\n[Step 3/3] Deploying jobs to Kubernetes...")
    deploy_jobs(job_files)
    
    print("\n" + "=" * 70)
    print("All done!")
    print("=" * 70)
    
    print(f"\nSweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    
    print(f"\nMonitor jobs:")
    print(f"\nMonitor jobs:")
    print(f"  kubectl get jobs -l app=wandb-sweep")
    print(f"\nCheck logs (examples):")
    for i in range(1, min(3, args.num + 1)):
        print(f"  kubectl logs -f job/mcscf-gnn-sweep-{i}")
    if args.num > 2:
        print(f"  ... (jobs 1-{args.num})")
    
    print(f"\nDelete all jobs:")
    print(f"  python manage_training.py --delete")

if __name__ == "__main__":
    main()
