import os
import hashlib
from datetime import datetime
from pathlib import Path

ALGO = 'sha256'
NUM_BACKUP = 5
LOG_FILE = "checksums.log"

def compute_checksum(file_path):
    """Compute checksum of a file."""
    h = hashlib.new(ALGO)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    return h.hexdigest()

def print_log(log_file_path):
    print("printing log: ")
    with open(log_file_path, 'r') as log:
        print(log.read())

def save_checkpoint(file_path):
    """Save file into storage with timestamp and log checksum."""
    file_path = Path(file_path)
    storage_dir = file_path.parent
    log_file = storage_dir / "checksums.log"

    # Ensure the directory exists
    storage_dir.mkdir(exist_ok=True)

    checksum = compute_checksum(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{file_path.name}"
    new_file_path = storage_dir / new_filename

    # Rename (move) file
    file_path.rename(new_file_path)

    with open(log_file, 'a+') as log:
        log.writelines(f"{new_filename},{checksum}\n")

    lines = []
    with open(log_file, 'r') as log:
        lines = log.readlines()
        lines = list(filter(lambda l : l.strip() != "", sorted(lines, reverse=True)))
        if len(lines) > NUM_BACKUP:
            num_rem = len(lines)-NUM_BACKUP
            to_remove = lines[-num_rem:]
            for line in to_remove:
                file_name, _ = line.strip().split(',')
                os.remove(storage_dir / file_name)
            lines = lines[:-num_rem]
    with open(log_file, 'w') as log:
        log.writelines(lines)

    print(f"Saved {file_path.name} as {new_filename}")
    return new_file_path, checksum

def get_checkpoint(file_path):
    """Find and return the latest file matching a checksum, verifying actual file checksum."""
    file_path = Path(file_path)
    storage_dir = file_path.parent
    log_file = storage_dir / LOG_FILE

    if not log_file.exists():
        print("WARNING: No log file found.")
        return None

    with open(log_file, 'r') as log:
        lines = log.readlines()
        for line in sorted(lines, reverse=True):
            tmp = line.strip()
            if tmp == "":
                continue
            tmp = tmp.split(',')
            filename, checksum = tmp
            new_file_path = storage_dir / filename
            if not os.path.isfile(new_file_path):
                print(f"WARNING: file {new_file_path} missing")
                continue
            computed_checksum = compute_checksum(new_file_path)
            if computed_checksum != checksum:
                print(f"WARNING: checksum mismatch between log and file, skipping")
                continue
            return new_file_path
    print(f"WARNING: no matching file found")
    return None

if __name__ == "__main__":
    # tests 
    import time
    cur_dir = Path.cwd()
    for i in range(100000):
        test_dir = cur_dir / f"test_dir{i}"
        if not os.path.exists(test_dir):
            break
    os.makedirs(test_dir)
    filename = "test_z9smon.txt"
    test_file = test_dir / filename
    def save_text(text):
        with open(test_file, 'w') as f:
            test_file.write_text(text)
    print("TEST: running save load tests")
    prev_path, new_path = None, None
    # save and load test
    for i in range(NUM_BACKUP*2):
        save_text(f"Hello 12{i}")
        prev_path = new_path
        new_path, c = save_checkpoint(test_file)
        time.sleep(2)
        print_log(test_dir/LOG_FILE)
        assert new_path == get_checkpoint(test_file)
        assert len([f for f in os.listdir(test_dir) if filename in f]) <= NUM_BACKUP
    
    print("TEST: running checksum integrity tests")
    # integrity test
    with open(new_path, 'w') as f:
        f.write("-------")
    assert prev_path == get_checkpoint(test_file)
    print("all passed")
    import shutil
    shutil.rmtree(test_dir)

