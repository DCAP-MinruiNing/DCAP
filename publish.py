import subprocess

def run_cmd(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.stdout.read(), process.stderr.read()
    if stdout: print(stdout)
    if stderr: print(stderr)
    return process.returncode

def main():
    print("=========================================")
    print("🚀 启动 DCAP 本地一键自动推送工具")
    print("=========================================")

    print("\n📦 正在将更新的文件打包到暂存区 (git add)...")
    if run_cmd("git add .") != 0:
        return

    print("\n📝 正在提交到本地仓库 (git commit)...")
    run_cmd('git commit -m "Auto-update: sync local targets and static files"')

    print("\n☁️ 正在推送到 GitHub 远程仓库 (git push origin main)...")
    if run_cmd("git push origin main") == 0:
        print("\n🎉 本地一键推送成功！线上网页马上更新。")
    else:
        print("\n❌ 推送失败，请检查网络或权限。")

if __name__ == "__main__":
    main()