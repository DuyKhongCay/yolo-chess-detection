"""Debug and fix Hailo SDK environment, binary permissions, and log paths."""

import os
import shutil
from pathlib import Path

# Fix USER env var
if "USER" not in os.environ:
    os.environ["USER"] = "root"


def fix_log_directory_conflict(log_dir: str = "logs"):
    """Remove hailort.log directory if created by mistake."""
    hailort_path = Path(log_dir) / "hailort.log"
    if hailort_path.is_dir():
        shutil.rmtree(hailort_path)
        print(f"[✓] Removed directory conflict at {hailort_path}")


def fix_binary_permissions():
    """Ensure all Hailo SDK C++ binary tools have executable (+x) permissions."""
    try:
        import hailo_sdk_client
        sdk_base = Path(hailo_sdk_client.__file__).parent
        fixed_count = 0
        for p in sdk_base.rglob("*"):
            if p.is_file() and not p.suffix and any(k in p.name.lower() for k in ["tool", "hailo", "builder", "allocator"]):
                p.chmod(p.stat().st_mode | 0o755)
                fixed_count += 1
        print(f"[✓] Granted execution permissions to {fixed_count} Hailo SDK binaries.")
    except Exception as e:
        print(f"[!] Failed to update permissions: {e}")


def inspect_hailo_runner():
    """Inspect HailoToolsRunner source code and methods."""
    try:
        import inspect
        import hailo_sdk_client.allocator.hailo_tools_runner as htr
        print("=== SOURCE OF run_tool_from_binary ===")
        print(inspect.getsource(htr.run_tool_from_binary))
        print("=== SOURCE OF run_hailo_tools ===")
        print(inspect.getsource(htr.run_hailo_tools))
    except Exception as e:
        print(f"[!] Inspection failed: {e}")


def fix_sdk_paths_dist_packages():
    """Patch Hailo SDKPaths to recognize dist-packages in Debian/Ubuntu environment."""
    try:
        import hailo_sdk_common
        paths_file = Path(hailo_sdk_common.__file__).parent / "paths_manager" / "paths.py"
        if paths_file.exists():
            lines = paths_file.read_text().splitlines()
            new_lines = []
            for line in lines:
                if "self._is_release =" in line:
                    new_lines.append(
                        '        self._is_release = any(pkg in os.path.dirname(os.path.dirname(hailo_sdk_common.origin)) for pkg in ["site-packages", "dist-packages"])'
                    )
                else:
                    new_lines.append(line)
            paths_file.write_text("\n".join(new_lines) + "\n")
            print(f"[✓] Cleanly patched {paths_file} for dist-packages.")
    except Exception as e:
        print(f"[!] Could not patch SDKPaths: {e}")


def main():
    """Run all environment setup and debug checks."""
    print("[+] Running Hailo SDK Environment Debug & Fix Script...")
    fix_log_directory_conflict()
    fix_binary_permissions()
    fix_sdk_paths_dist_packages()
    print("[✓] Environment check completed.")


if __name__ == "__main__":
    main()
