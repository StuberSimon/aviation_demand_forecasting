def main():
    import subprocess
    from datetime import datetime
    import time

    time.sleep(6)

    print(f"\n\nStarting Train_model_script_v1.py at {datetime.now()}\n\n")
    subprocess.Popen(["python", "Train_model_script_v1.py"])


if __name__ == '__main__':
    main()