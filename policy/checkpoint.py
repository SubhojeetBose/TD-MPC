import os
import hashlib
from datetime import datetime
from pathlib import Path

ALGO = 'sha256'
LOG_FILE = "checksums.log"

def get_logfile_path(file_path):
    storage_dir = file_path.parent
    log_file = storage_dir / f"{file_path.stem}_checksums.log"
    return log_file

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

def save_checkpoint(file_path, num_backup=-1):
    """Save file into storage with timestamp and log checksum."""
    file_path = Path(file_path)
    storage_dir = file_path.parent
    log_file = get_logfile_path(file_path)

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

    # remove older checkpoint files
    if num_backup > 0:
        lines = []
        with open(log_file, 'r') as log:
            lines = log.readlines()
            lines = list(filter(lambda l : l.strip() != "", sorted(lines, reverse=True)))
            if len(lines) > num_backup:
                num_rem = len(lines)-num_backup
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
    log_file = get_logfile_path(file_path)

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
    TEST_NUM_BACKUP = 5
    SLEEP_TIME = 1.1
    filename = "test.txt"
    test_file = test_dir / filename
    def save_text(test_file, text):
        with open(test_file, 'w') as f:
            test_file.write_text(text)

    def validate_log(file_path):
        # for each entry in log file, that file should exist with same checkpoint
        storage_dir = file_path.parent
        log_file = get_logfile_path(file_path)

        with open(log_file, 'r') as log:
            lines = log.readlines()
        for line in lines:
            tmp = line.strip()
            if tmp == "":
                continue
            tmp = tmp.split(',')
            filename, checksum = tmp
            new_file_path = storage_dir / filename
            assert os.path.isfile(new_file_path)
            computed_checksum = compute_checksum(new_file_path)
            assert computed_checksum == checksum
    print("TEST: running save load tests")
    checkpoint_files = []
    # save and load test
    for i in range(TEST_NUM_BACKUP*2):
        save_text(test_file, f"Hello 12{i}")
        new_path, c = save_checkpoint(test_file, TEST_NUM_BACKUP)
        checkpoint_files.append(new_path)
        actual_path = get_checkpoint(test_file)
        assert str(new_path) == str(actual_path), f"assertion failed expected path: {new_path}, actual: {actual_path}"
        assert len([f for f in os.listdir(test_dir) if filename in f]) <= TEST_NUM_BACKUP
        assert c == compute_checksum(new_path)
        validate_log(test_file)
        time.sleep(SLEEP_TIME)
    
    print("TEST: running checksum integrity tests")
    # integrity test
    checkpoint_files.reverse()
    get_idx = 1
    for file in checkpoint_files[:TEST_NUM_BACKUP-1]:
        with open(file, 'w') as f:
            f.write("-------")
        actual = str(get_checkpoint(test_file))
        expected = str(checkpoint_files[get_idx])
        assert expected == actual, f"assertion failed expected path: {expected}, actual: {actual}"
        get_idx += 1

    print("TEST: no delete")
    checkpoint_test_file = test_file.parent / "checkpoint.txt"
    for i in range(TEST_NUM_BACKUP*2):
        save_text(checkpoint_test_file, f"Hello 12{i}")
        save_checkpoint(checkpoint_test_file)
        validate_log(checkpoint_test_file)
        time.sleep(SLEEP_TIME)
    import shutil
    shutil.rmtree(test_dir)

    print("all passed")

