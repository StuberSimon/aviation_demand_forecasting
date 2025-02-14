def main():
    import subprocess
    from datetime import datetime
    import time

    time.sleep(6)

    print(f"\n\nStarting sbs_model_train_script.py at {datetime.now()}\n\n")
    subprocess.Popen(["python", "sbs_model_train_script.py"])


if __name__ == '__main__':
    main()