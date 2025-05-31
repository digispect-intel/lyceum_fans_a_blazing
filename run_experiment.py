import asyncio
import subprocess
import os
import sys
import argparse
from pathlib import Path

async def run_command(cmd, shell=True, stream_output=True):
    """Run a command asynchronously with detailed logging"""
    print(f"🚀 Running: {cmd}")
    
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=shell,
        limit=64*1024*1024
    )
    
    if stream_output:
        # Stream output in real-time with better error handling
        async def read_stream(stream, prefix):
            try:
                while True:
                    try:
                        line = await stream.readline()
                        if not line:
                            break
                        # Truncate very long lines
                        line_str = line.decode().rstrip()
                        if len(line_str) > 500:
                            line_str = line_str[:500] + "... [truncated]"
                        print(f"{prefix} {line_str}")
                    except asyncio.LimitOverrunError:
                        # Handle lines that are too long
                        print(f"{prefix} [Line too long - skipped]")
                        # Read and discard the problematic line
                        try:
                            await stream.readuntil(b'\n')
                        except:
                            break
            except Exception as e:
                print(f"{prefix} Stream reading error: {e}")
        
        # Start both stdout and stderr readers
        try:
            await asyncio.gather(
                read_stream(process.stdout, "📤"),
                read_stream(process.stderr, "⚠️ "),
                process.wait()
            )
        except Exception as e:
            print(f"⚠️  Stream processing error: {e}")
        
        returncode = process.returncode
        stdout_data = ""
        stderr_data = ""
    else:
        stdout_data, stderr_data = await process.communicate()
        returncode = process.returncode
        
        if stdout_data:
            print(f"📤 Output: {stdout_data.decode()}")
        if stderr_data:
            print(f"⚠️  Error output: {stderr_data.decode()}")
    
    if returncode != 0:
        print(f"❌ Command failed with exit code {returncode}: {cmd}")
        return False, stderr_data.decode() if not stream_output else ""
    
    print(f"✅ Command completed successfully: {cmd}")
    return True, stdout_data.decode() if not stream_output else ""

def parse_config_file(config_path):
    """Extract run name and model info from dstack config file"""
    print(f"📋 Parsing config file: {config_path}")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    print(f"📄 Config file contents:")
    for i, line in enumerate(content.split('\n'), 1):
        print(f"   {i:2d}: {line}")
    
    # Extract run name
    run_name = None
    for line in content.split('\n'):
        if line.strip().startswith('name:'):
            run_name = line.split(':')[1].strip()
            break
    
    if not run_name:
        raise ValueError(f"Could not find 'name:' field in {config_path}")
    
    print(f"🏷️  Extracted run name: {run_name}")
    return run_name

def get_model_info_from_path(config_path):
    """Extract model category and name from file path"""
    print(f"📁 Analyzing path: {config_path}")
    path_parts = Path(config_path).parts
    print(f"🔍 Path parts: {path_parts}")
    
    # Expected structure: models/category/model_name/config.dstack.yml
    if len(path_parts) >= 3 and path_parts[0] == 'models':
        category = path_parts[1]  # e.g., 'text_generation'
        model_name = path_parts[2]  # e.g., 'gpt2'
        print(f"📊 Model category: {category}")
        print(f"🤖 Model name: {model_name}")
        return category, model_name
    else:
        raise ValueError(f"Config path doesn't match expected structure: {config_path}")

async def wait_for_ssh_window(run_name):
    """Wait for dev environment to be ready and experiment to complete"""
    print(f"⏳ Waiting for dev environment {run_name} to be ready...")
    
    attempt = 1
    while True:
        print(f"🔍 Checking logs (attempt {attempt})...")
        success, output = await run_command(f"dstack logs {run_name}", stream_output=False)
        
        if success:
            # For dev environments, look for experiment completion
            if "Test completed successfully!" in output:
                print("🎉 Experiment completed! SSH should be available...")
                break
            elif "To open in VS Code Desktop, use this link:" in output:
                print("🚪 Dev environment ready! Checking if experiment completed...")
                # Environment is ready, but experiment might still be running
                if "Test completed successfully!" in output:
                    break
        
        print(f"⏰ Waiting 10 seconds before next check...")
        await asyncio.sleep(10)
        attempt += 1
        
        if attempt > 60:  # 10 minute timeout
            print("⏰ Timeout - proceeding with download attempt...")
            break

async def download_file_via_dstack_automation(run_name, remote_path, local_path):
    """Fully automated file download using dstack attach + scp"""
    print(f"📥 Starting automated file download...")
    print(f"   Run name: {run_name}")
    print(f"   Remote: {remote_path}")
    print(f"   Local: {local_path}")
    
    # Step 1: Start dstack attach in background (keeps SSH alive)
    print(f"🔗 Starting SSH connection...")
    attach_cmd = f"dstack attach {run_name}"
    
    attach_process = await asyncio.create_subprocess_shell(
        attach_cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    
    # Step 2: Wait for SSH connection to establish
    print(f"⏳ Waiting for SSH connection to establish...")
    await asyncio.sleep(10)  # Give more time for connection
    
    # Step 3: Attempt SCP download with retries
    max_retries = 3
    for attempt in range(max_retries):
        print(f"📁 Download attempt {attempt + 1}/{max_retries}...")
        
        scp_cmd = f"scp -o ConnectTimeout=30 -o StrictHostKeyChecking=no {run_name}:{remote_path} {local_path}"
        
        scp_process = await asyncio.create_subprocess_shell(
            scp_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await scp_process.communicate()
        
        if scp_process.returncode == 0:
            print(f"✅ SCP download successful!")
            break
        else:
            print(f"⚠️  SCP attempt {attempt + 1} failed:")
            print(f"   Return code: {scp_process.returncode}")
            print(f"   Error: {stderr.decode()}")
            
            if attempt < max_retries - 1:
                print(f"   Retrying in 5 seconds...")
                await asyncio.sleep(5)
    else:
        print("❌ All SCP attempts failed")
        # Clean up attach process
        try:
            attach_process.terminate()
            await attach_process.wait()
        except:
            pass
        return False
    
    # Step 4: Clean up attach process
    try:
        attach_process.terminate()
        await attach_process.wait()
    except:
        pass
    
    # Step 5: Verify download
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        file_size = os.path.getsize(local_path)
        print(f"✅ File downloaded successfully: {local_path} ({file_size} bytes)")
        return True
    else:
        print("❌ File download verification failed")
        return False

async def download_file_simple(run_name, remote_path, local_path):
    """Simple direct SCP download - mirrors manual process"""
    print(f"📥 Downloading {remote_path} from {run_name}...")
    
    # Direct SCP command (same as manual)
    scp_cmd = f"scp {run_name}:{remote_path} {local_path}"
    
    success, output = await run_command(scp_cmd, stream_output=False)
    
    if success and os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        print(f"✅ Downloaded: {local_path} ({file_size} bytes)")
        return True
    else:
        print(f"❌ Download failed: {output}")
        return False


async def run_experiment(config_path, result_filename=None):
    """Run the complete experiment workflow"""
    
    print(f"🎯 Starting experiment workflow")
    print(f"=" * 50)
    
    # Parse configuration
    run_name = parse_config_file(config_path)
    category, model_name = get_model_info_from_path(config_path)
    
    # Default result filename if not provided
    if not result_filename:
        result_filename = "results.parquet"
    
    print(f"📋 Experiment Summary:")
    print(f"   Config: {config_path}")
    print(f"   Run name: {run_name}")
    print(f"   Model: {category}/{model_name}")
    print(f"   Expected result file: {result_filename}")
    print(f"=" * 50)
    
    # Step 1: Apply the dstack configuration with -y flag
    print("🚀 Step 1: Applying dstack configuration...")
    success, output = await run_command(f"dstack apply -y -f {config_path}")
    if not success:
        print("❌ Failed to apply dstack configuration")
        return False
    
    # Step 2: Wait for SSH window
    print("⏳ Step 2: Waiting for SSH window...")
    await wait_for_ssh_window(run_name)
    
    # Step 3: Create local directory
    local_dir = f"./data/{category}/{model_name}/"
    print(f"📁 Step 3: Creating local directory: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    print(f"✅ Directory created/verified: {local_dir}")
    
    # Step 4: Download the file via scp
    machine_type = run_name.replace(f"{model_name}-", "").replace("-test", "")

    remote_path = f"/workflow/{result_filename}"
    local_filename = f"{model_name}_{machine_type}_results.parquet"
    local_path = f"{local_dir}{local_filename}"

    success = await download_file_simple(run_name, remote_path, local_path)
    if not success:
        print("❌ Failed to download file")
        return False

    # Step 5: Capture dstack metrics
    print("📊 Step 5: Capturing dstack metrics...")
    metrics_success, metrics_output = await run_command(f"dstack metrics {run_name}", stream_output=False)
    
    if metrics_success:
        # Save metrics to file
        metrics_filename = f"{model_name}_{machine_type}_metrics.txt"
        metrics_path = f"{local_dir}{metrics_filename}"
        
        with open(metrics_path, 'w') as f:
            f.write(metrics_output)
        
        print(f"✅ Metrics saved to: {metrics_path}")
    else:
        print("⚠️  Failed to capture dstack metrics")

    print("🎉 Experiment completed successfully!")
    print(f"💾 Results saved to: {local_path} and {metrics_path}")
    
    return True

async def main():
    """Main async function"""
    parser = argparse.ArgumentParser(description='Run dstack experiments and download results')
    parser.add_argument('config_path', help='Path to dstack configuration file')
    parser.add_argument('--result-file', help='Name of result file to download (default: auto-detect)')
    
    args = parser.parse_args()
    
    print(f"🏁 Starting dstack experiment runner")
    
    try:
        success = await run_experiment(args.config_path, args.result_file)
        if success:
            print("🎉 All steps completed successfully!")
        else:
            print("💥 Experiment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
