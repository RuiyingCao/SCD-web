from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for, jsonify
import os
import subprocess
import glob
import numpy as np
import time
import uuid
import threading

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 用于加密 session，生产环境中应使用安全的密钥

UPLOAD_FOLDER = "uploads"
TASKS = {}
TASKS_LOCK = threading.Lock()

def create_task():
    task_id = uuid.uuid4().hex
    with TASKS_LOCK:
        TASKS[task_id] = {
            "state": "running",
            "progress": 0,
            "message": "Preparing...",
            "result": None
        }
    return task_id

def set_task(task_id, *, state=None, progress=None, message=None, result=None):
    with TASKS_LOCK:
        if task_id not in TASKS:
            return
        if state is not None:
            TASKS[task_id]["state"] = state
        if progress is not None:
            TASKS[task_id]["progress"] = progress
        if message is not None:
            TASKS[task_id]["message"] = message
        if result is not None:
            TASKS[task_id]["result"] = result

def parse_nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]

@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({"state": "error", "message": "Task not found"}), 404
    return jsonify(task)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file_type = request.form.get("fileType")
        t = None

        if file_type == "sequencing":
            file = request.files.get("file")
            R = request.form.get("R", 2)
            preset_coverage = request.form.get("preset_coverage", 5)
            expected_completeness = request.form.get("expected_completeness", 0.5)
            mark_option = request.form.get("mark_option", "preset_coverage")

            if file:
                try:
                    R = float(R)
                    if R < 1.1:
                        return "The minimum value of parameter R is 1.1！", 400
                except ValueError:
                    return "The parameter R must be a numerical value！", 400

                try:
                    preset_coverage = float(preset_coverage)
                    if preset_coverage < 0.5 or preset_coverage > 30:
                        return "The parameter Preset sequencing coverage depth must be between 0.5 and 10！"
                    if preset_coverage % 0.5 != 0:
                        return "The parameter Preset sequencing coverage depth must be a multiple of 0.5！"
                except ValueError:
                    return "The parameter Preset sequencing coverage depth must be a numerical value！", 400

                try:
                    expected_completeness = float(expected_completeness)
                    if expected_completeness < 0 or expected_completeness > 1:
                        return "The parameter Expected completeness of sample information extraction must be between 0 and 1！"
                except ValueError:
                    return "The parameter Expected completeness of sample information extraction must be a numerical value！", 400

                for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
                    os.remove(old_image)

                file_name = file.filename
                file_path = os.path.join(UPLOAD_FOLDER, file_name)
                file.save(file_path)
                session["file_path"] = file_path  # 确保文件路径存储在会话中
                session["file_name"] = file_name
                session["R"] = R
                session["preset_coverage"] = preset_coverage
                session["expected_completeness"] = expected_completeness
                session["mark_option"] = mark_option
                session["R_noisy"] = 2
                session["a"] = 2
                session["c"] = 0.01
                # 不保存 t 到 session，因为对于 sequencing 不需要 t

                scripts = ["1_seqnum_direct.py", "2_seqnum_log.py", "3_seqpro_log.py", "4_population dist.py"]
                for script in scripts:
                    try:
                        if script == "4_population dist.py":
                            result = subprocess.run(
                                [
                                    "python",
                                    script,
                                    file_path,
                                    str(R),
                                    str(preset_coverage),
                                    str(expected_completeness),
                                    mark_option
                                ],
                                check=True,
                                cwd=os.getcwd(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8'
                            )
                        else:
                            result = subprocess.run(
                                ["python", script, file_path],
                                check=True,
                                cwd=os.getcwd(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8'
                            )
                    except subprocess.CalledProcessError as e:
                        return f"fail to run {script}: {e.stderr}", 500

                # 设置默认值
                session["R_noisy"] = 2
                session["a"] = 2
                session["c"] = 0.01

                # 运行 5_bounds.py 计算上下界
                try:
                    result = subprocess.run(
                        ["python", "5_bounds.py", file_path, str(session["R_noisy"]), str(session["c"]),
                         str(session["a"])],
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=os.getcwd(),
                        encoding='utf-8'
                    )
                    lines = result.stdout.strip().split('\n')
                    lower_bound, center_bound, upper_bound = map(float, lines[:3])
                    print(f"5_bounds.py output: {result.stdout}")  # 调试信息
                except subprocess.CalledProcessError as e:
                    return f"fail to run 5_bounds.py: {e.stderr}", 500

                session["lower_bound"] = lower_bound
                session["upper_bound"] = upper_bound
                session["center_bound"] = center_bound

                return redirect(url_for("result"))

        elif file_type == "pcr_efficiency" or file_type == "simulate":
            if file_type == "pcr_efficiency":
                pcr_efficiency_file = request.files.get("pcr_efficiency_file")
                x = request.form.get("x")
                y = request.form.get("y")
                option = request.form.get("option")
                S = request.form.get("S") if option == "total_sequences" else None
                t = request.form.get("t_pcr_efficiency")  # 获取 t 值
                print(f"pcr_efficiency received t value: {t}")
                session["t"] = t

                if pcr_efficiency_file:
                    try:
                        x = float(x)
                        y = float(y)
                        if S:
                            S = int(S)
                    except ValueError:
                        return "The parameter must be a number！", 400
                    for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
                        os.remove(old_image)
                    pcr_efficiency_file_name = pcr_efficiency_file.filename
                    pcr_efficiency_file_path = os.path.join(UPLOAD_FOLDER, pcr_efficiency_file_name)
                    pcr_efficiency_file.save(pcr_efficiency_file_path)
                    session["pcr_efficiency_file_path"] = pcr_efficiency_file_path  # 确保文件路径存储在会话中
                    session["x"] = x
                    session["y"] = y
                    session["option"] = option
                    session["S"] = S

                    # 生成合成条数文件
                    synthesisnum_file_path = os.path.join(UPLOAD_FOLDER, "synthesisnum.txt")
                    if option == "per_sequence":
                        # 调用 6.1.1_generate_ci.py
                        try:
                            result = subprocess.run(
                                ["python", "6.1.1_generate_ci.py", str(x), str(y), pcr_efficiency_file_path],
                                check=True,
                                cwd=os.getcwd(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8'
                            )
                            print(f"6.1.1_generate_ci.py output: {result.stdout}")  # 调试信息
                        except subprocess.CalledProcessError as e:
                            return f"fail to run 6.1.1_generate_ci.py: {e.stderr}", 500
                    elif option == "total_sequences":
                        if not S:
                            return "The parameter S must be filled in！", 400
                        # 调用 6.1.2_generate_ci(整体).py
                        try:
                            result = subprocess.run(
                                ["python", "6.1.2_generate_ci(整体).py", str(S), pcr_efficiency_file_path],
                                check=True,
                                cwd=os.getcwd(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8'
                            )
                            print(f"6.1.2_generate_ci(整体).py output: {result.stdout}")  # 调试信息
                        except subprocess.CalledProcessError as e:
                            return f"fail to run 6.1.2_generate_ci(整体).py: {e.stderr}", 500

                    # 检查生成的 synthesisnum.txt 行数
                    try:
                        with open(synthesisnum_file_path, 'r') as f:
                            synthesisnum_lines = sum(1 for line in f)
                        with open(pcr_efficiency_file_path, 'r') as f:
                            pcr_efficiency_lines = sum(1 for line in f)
                        print(
                            f"synthesisnum.txt lines: {synthesisnum_lines}, PCR_efficiency.txt lines: {pcr_efficiency_lines}")
                        if synthesisnum_lines != pcr_efficiency_lines:
                            return f"生成的 synthesisnum.txt 行数 ({synthesisnum_lines}) 与 PCR_efficiency.txt 行数 ({pcr_efficiency_lines}) 不一致！", 500
                    except Exception as e:
                        return f"An error occurred while reading the number of lines in the file: {str(e)}", 500

                    # 运行分析脚本
                    if 'synthesisnum_file_path' in locals() and synthesisnum_file_path:
                        try:
                            args = [
                                "python",
                                "7_analysis.py",
                                synthesisnum_file_path,
                                pcr_efficiency_file_path
                            ]
                            if t is not None:
                                args.extend(["--t", str(t)])  # 使用 --t 前缀传递 t 值
                            result = subprocess.run(
                                args,
                                check=True,
                                cwd=os.getcwd(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8'
                            )
                            mu_t, sigma_t = map(float, result.stdout.strip().split('\n'))
                            print(f"7_analysis.py output: {result.stdout}")  # 调试信息
                        except subprocess.CalledProcessError as e:
                            return f"fail to run 7_analysis.py: {e.stderr}", 500
                    else:
                        return "生成的 synthesisnum.txt 文件路径未定义！", 500

                    # 设置默认值
                    session["R_noisy"] = 2
                    session["a"] = 2
                    session["c"] = 0.01
                    # 保存 mu_t 和 sigma_t 到 session
                    session["mu_t"] = mu_t
                    session["sigma_t"] = sigma_t
                    # 计算上下界
                    try:
                        result = subprocess.run(
                            [
                                "python",
                                "8_bounds.py",
                                synthesisnum_file_path,
                                str(session.get("R_noisy")),
                                str(session.get("c")),
                                str(session.get("a")),
                                str(mu_t),
                                str(sigma_t)
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                            cwd=os.getcwd(),
                            encoding='utf-8'
                        )
                        lines = result.stdout.strip().split('\n')
                        lower_bound, center_bound, upper_bound = map(float, lines[:3])
                        print(f"8_bounds.py output: {result.stdout}")  # 调试信息
                    except subprocess.CalledProcessError as e:
                        return f"fail to run 8_bounds.py: {e.stderr}", 500

                    session["lower_bound"] = lower_bound
                    session["upper_bound"] = upper_bound
                    session["center_bound"] = center_bound
                    # 保存 file_path 到 session
                    session["file_path"] = synthesisnum_file_path
                    return redirect(url_for("result2"))

            elif file_type == "simulate":
                n = request.form.get("n")
                simulate_x = request.form.get("simulateX")
                simulate_y = request.form.get("simulateY")
                simulate_option = request.form.get("simulateOption")
                simulate_x_per_sequence = request.form.get("simulateXPerSequence")
                simulate_y_per_sequence = request.form.get("simulateYPerSequence")
                S = request.form.get("simulateS")
                t = request.form.get("t_simulate")  # 获取 t 值
                print(f"simulate received t value: {t}")
                session["t"] = t

                if not n or not simulate_x or not simulate_y:
                    return "参数 n, simulateX, simulateY 必须填写！", 400

                try:
                    n = int(n)
                    simulate_x = float(simulate_x)
                    simulate_y = float(simulate_y)
                    simulate_x_per_sequence = float(simulate_x_per_sequence)
                    simulate_y_per_sequence = float(simulate_y_per_sequence)
                    if S:
                        S = int(S)
                except ValueError:
                    return "参数必须是数字！", 400

                for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
                    os.remove(old_image)

                # 生成 PCR_efficiency.txt
                try:
                    pcr_efficiency_file_path = os.path.join(UPLOAD_FOLDER, "PCR_efficiency.txt")
                    result = subprocess.run(
                        ["python", "6.2_generate_ri.py", str(n), str(simulate_x), str(simulate_y)],
                        check=True,
                        cwd=os.getcwd(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8'
                    )
                    print(f"6.2_generate_ri.py output: {result.stdout}")  # 调试信息
                except subprocess.CalledProcessError as e:
                    return f"运行 6.2_generate_ri.py 失败: {e.stderr}", 500

                # 生成 synthesisnum.txt
                synthesisnum_file_path = os.path.join(UPLOAD_FOLDER, "synthesisnum.txt")
                if simulate_option == "per_sequence":
                    try:
                        result = subprocess.run(
                            ["python", "9.1_ci.py", str(n), str(simulate_x_per_sequence), str(simulate_y_per_sequence)],
                            check=True,
                            cwd=os.getcwd(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8'
                        )
                        print(f"9.1_ci.py output: {result.stdout}")  # 调试信息
                    except subprocess.CalledProcessError as e:
                        return f"运行 9.1_ci.py 失败: {e.stderr}", 500
                elif simulate_option == "total_sequences":
                    if not S:
                        return "参数 S 必须填写！", 400
                    # 调用 9.2_ci.py
                    try:
                        result = subprocess.run(
                            ["python", "9.2_ci.py", str(n), str(S)],
                            check=True,
                            cwd=os.getcwd(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8'
                        )
                        print(f"9.2_ci.py output: {result.stdout}")  # 调试信息
                    except subprocess.CalledProcessError as e:
                        return f"运行 9.2_ci.py 失败: {e.stderr}", 500

                # 检查生成的 synthesisnum.txt 行数
                try:
                    with open(synthesisnum_file_path, 'r') as f:
                        synthesisnum_lines = sum(1 for line in f)
                    print(f"synthesisnum.txt 行数: {synthesisnum_lines}")
                except Exception as e:
                    return f"读取文件行数时出错: {str(e)}", 500

                # 运行分析脚本
                if 'synthesisnum_file_path' in locals() and synthesisnum_file_path:
                    try:
                        args = [
                            "python",
                            "7_analysis.py",
                            synthesisnum_file_path,
                            pcr_efficiency_file_path
                        ]
                        if t is not None:
                            args.extend(["--t", str(t)])  # 使用 --t 前缀传递 t 值
                        result = subprocess.run(
                            args,
                            check=True,
                            cwd=os.getcwd(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8'
                        )
                        mu_t, sigma_t = map(float, result.stdout.strip().split('\n'))
                        print(f"7_analysis.py output: {result.stdout}")  # 调试信息
                    except subprocess.CalledProcessError as e:
                        return f"运行 7_analysis.py 失败: {e.stderr}", 500
                else:
                    return "生成的 synthesisnum.txt 文件路径未定义！", 500

                # 设置默认值
                session["R_noisy"] = 2
                session["a"] = 2
                session["c"] = 0.01

                # 保存 mu_t 和 sigma_t 到 session
                session["mu_t"] = mu_t
                session["sigma_t"] = sigma_t

                # 计算上下界
                try:
                    result = subprocess.run(
                        [
                            "python",
                            "8_bounds.py",
                            synthesisnum_file_path,
                            str(session.get("R_noisy")),
                            str(session.get("c")),
                            str(session.get("a")),
                            str(mu_t),
                            str(sigma_t)
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=os.getcwd(),
                        encoding='utf-8'
                    )
                    lines = result.stdout.strip().split('\n')
                    lower_bound, center_bound, upper_bound = map(float, lines[:3])
                    print(f"8_bounds.py output: {result.stdout}")  # 调试信息
                except subprocess.CalledProcessError as e:
                    return f"运行 8_bounds.py 失败: {e.stderr}", 500

                session["lower_bound"] = lower_bound
                session["upper_bound"] = upper_bound
                session["center_bound"] = center_bound

                # 保存 file_path 到 session
                session["file_path"] = synthesisnum_file_path

                return redirect(url_for("result3"))

            # 验证 t 值
            if t is None:
                return "参数 t 必须填写！", 400
            try:
                t = int(t)
                if t < 1:
                    return "参数 t 的最小值为 1！", 400
            except ValueError:
                return "参数 t 必须是整数！", 400

    return render_template("index.html", file_uploaded=False, file_name="", lower_bound=None, upper_bound=None, R=2, c=0.01, a=2, expected_completeness=0.5, mark_option="preset_coverage")


@app.route("/result", methods=["GET"])
def result():
    file_name = session.get("file_name", "")
    file_path = session.get("file_path", "")
    lower_bound = session.get("lower_bound", None)
    upper_bound = session.get("upper_bound", None)
    center_bound = session.get("center_bound", None)
    R = session.get("R", 2)
    c = session.get("c", 0.01)
    a = session.get("a", 2)
    preset_coverage = session.get("preset_coverage", 5)
    R_noisy = session.get("R_noisy", 2)
    expected_completeness = session.get("expected_completeness", 0.5)
    mark_option = session.get("mark_option", "preset_coverage")

    return render_template(
        "result.html",
        file_uploaded=file_path is not None,
        file_name=file_name,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        center_bound=center_bound,
        R=R,
        c=c,
        a=a,
        preset_coverage=preset_coverage,
        R_noisy=R_noisy,
        expected_completeness=expected_completeness,
        mark_option=mark_option
    )


@app.route("/result2", methods=["GET"])
def result2():
    file_name = session.get("file_name", "")
    lower_bound = session.get("lower_bound", None)
    upper_bound = session.get("upper_bound", None)
    center_bound = session.get("center_bound", None)
    R = session.get("R", 2)
    c = session.get("c", 0.01)
    a = session.get("a", 2)
    preset_coverage = session.get("preset_coverage", 5)
    R_noisy = session.get("R_noisy", 2)
    expected_completeness = session.get("expected_completeness", 0.5)
    mark_option = session.get("mark_option", "preset_coverage")

    return render_template(
        "result2.html",
        file_uploaded=file_name is not None,
        file_name=file_name,
        lower_bound=lower_bound,
        center_bound=center_bound,
        upper_bound=upper_bound,
        R=R,
        c=c,
        a=a,
        preset_coverage=preset_coverage,
        R_noisy=R_noisy,
        expected_completeness=expected_completeness,
        mark_option=mark_option
    )


@app.route("/result3", methods=["GET"])
def result3():
    lower_bound = session.get("lower_bound", None)
    upper_bound = session.get("upper_bound", None)
    center_bound = session.get("center_bound", None)

    return render_template(
        "result3.html",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        center_bound=center_bound
    )


@app.route("/update_preset_coverage", methods=["POST"])
def update_preset_coverage():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    preset_coverage = request.form.get("preset_coverage", session.get("preset_coverage", 5))
    expected_completeness = request.form.get("expected_completeness", session.get("expected_completeness", 0.5))
    mark_option = request.form.get("mark_option", session.get("mark_option", "preset_coverage"))

    try:
        preset_coverage = float(preset_coverage)
        if preset_coverage < 0.5 or preset_coverage > 30:
            return jsonify({"success": False, "message": "参数 Preset sequencing coverage depth 必须在 0.5 到 10 之间！"}), 400
        if preset_coverage % 0.5 != 0:
            return jsonify({"success": False, "message": "参数 Preset sequencing coverage depth 必须是 0.5 的倍数！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 Preset sequencing coverage depth 必须是数字！"}), 400

    try:
        expected_completeness = float(expected_completeness)
        if expected_completeness < 0 or expected_completeness > 1:
            return jsonify({"success": False,
                            "message": "参数 Expected completeness of sample information extraction 必须在 0 到 1 之间！"}), 400
    except ValueError:
        return jsonify(
            {"success": False, "message": "参数 Expected completeness of sample information extraction 必须是数字！"}), 400

    # 删除旧的图片文件
    for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图4-1.png")):
        os.remove(old_image)

    # 保存新的 preset_coverage, expected_completeness, mark_option 值到 session
    session["preset_coverage"] = preset_coverage
    session["expected_completeness"] = expected_completeness
    session["mark_option"] = mark_option

    # 重新运行 4_population dist.py 并捕获坐标信息
    try:
        result = subprocess.run(
            [
                "python",
                "4_population dist.py",
                file_path,
                str(session.get("R")),
                str(preset_coverage),
                str(expected_completeness),
                mark_option
            ],
            check=True,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        coordinate_info_4_1, coordinate_info_4_2 = result.stdout.strip().split('\n')  # 分割返回的两个坐标信息
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "message": f"运行 4_population dist.py 失败: {e.stderr}"}), 500

    return jsonify({"success": True, "message": "Update successfully！", "coordinate_info_4_1": coordinate_info_4_1,
                    "coordinate_info_4_2": coordinate_info_4_2})  # 返回两个坐标信息


@app.route("/update_R", methods=["POST"])
def update_R():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    R = request.form.get("R", session.get("R", 2))
    preset_coverage = session.get("preset_coverage", 5)
    expected_completeness = session.get("expected_completeness", 0.5)
    mark_option = session.get("mark_option", "preset_coverage")

    try:
        R = float(R)
        if R < 1.1:
            return jsonify({"success": False, "message": "参数 R 的最小值为 1.1！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 R 必须是数字！"}), 400

    try:
        expected_completeness = float(expected_completeness)
        if expected_completeness < 0 or expected_completeness > 1:
            return jsonify({"success": False,
                            "message": "参数 Expected completeness of sample information extraction 必须在 0 到 1 之间！"}), 400
    except ValueError:
        return jsonify(
            {"success": False, "message": "参数 Expected completeness of sample information extraction 必须是数字！"}), 400

    # 删除旧的图片文件
    for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图4-2.png")):
        os.remove(old_image)

    # 保存新的 R, expected_completeness, mark_option 值到 session
    session["R"] = R
    session["expected_completeness"] = expected_completeness
    session["mark_option"] = mark_option

    # 重新运行 4_population dist.py 并捕获坐标信息
    try:
        result = subprocess.run(
            [
                "python",
                "4_population dist.py",
                file_path,
                str(R),
                str(preset_coverage),
                str(expected_completeness),
                mark_option
            ],
            check=True,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        coordinate_info_4_1, coordinate_info_4_2 = result.stdout.strip().split('\n')  # 分割返回的两个坐标信息
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "message": f"运行 4_population dist.py 失败: {e.stderr}"}), 500

    return jsonify(
        {"success": True, "message": "Update Successfully！", "coordinate_info_4_2": coordinate_info_4_2})  # 只返回 Fig.4-2 的坐标信息


@app.route("/calculate", methods=["POST"])
def calculate():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    R_noisy = request.form.get("R_noisy", session.get("R_noisy", 2))
    a = request.form.get("a", session.get("a", 2))
    c = 0.01  # 直接设置默认值

    try:
        R_noisy = float(R_noisy)
        if R_noisy < 1.1:
            return jsonify({"success": False, "message": "参数 R 的最小值为 1.1！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 R 必须是数字！"}), 400

    try:
        a = int(a)
        if a < 2:
            return jsonify({"success": False, "message": "参数 a 的最小值为 2！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 a 必须是整数！"}), 400

    try:
        c = float(c)
        if c <= 0:
            return jsonify({"success": False, "message": "参数 c 必须是正数！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 c 必须是数字！"}), 400

    # 删除旧的图片文件
    for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
        os.remove(old_image)

    # 保存新的 R_noisy, a, c 值到 session
    session["R_noisy"] = R_noisy
    session["a"] = a
    session["c"] = c

    # 运行 5_bounds.py 计算上下界
    try:
        result = subprocess.run(
            ["python", "5_bounds.py", file_path, str(R_noisy), str(c), str(a)],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
            encoding='utf-8'
        )
        lines = result.stdout.strip().split('\n')
        lower_bound, center_bound, upper_bound = map(float, lines[:3])
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "message": f"运行 5_bounds.py 失败: {e.stderr}"}), 500

    # 保存上下界到 session
    session["lower_bound"] = lower_bound
    session["upper_bound"] = upper_bound
    session["center_bound"] = center_bound

    return jsonify({"success": True, "message": "Update Successfully！", "lower_bound": lower_bound, "center_bound": center_bound, "upper_bound": upper_bound})


import importlib.util

# 动态导入 7_analysis.py
spec = importlib.util.spec_from_file_location("module.name", "7_analysis.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/update_parameter", methods=["POST"])
def update_parameter():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    pcr_efficiency_file_path = session.get("pcr_efficiency_file_path")
    if not pcr_efficiency_file_path:
        return jsonify({"success": False, "message": "未上传 PCR efficiency 文件！"}), 400

    parameter_type = request.json.get("parameter_type")
    parameter_value = request.json.get("parameter_value")
    coding_redundancy = request.json.get("coding_redundancy")
    t = session.get("t")  # 从 session 获取 t 值

    try:
        parameter_value = float(parameter_value)
    except ValueError:
        return jsonify({"success": False, "message": "参数必须是数字！"}), 400

    if parameter_type == "preset_coverage":
        if parameter_value < 0.5 or parameter_value > 30:
            return jsonify({"success": False, "message": "参数 Preset coverage depth 必须在 0.5 到 30 之间！"}), 400
        if parameter_value % 0.5 != 0:
            return jsonify({"success": False, "message": "参数 Preset coverage depth 必须是 0.5 的倍数！"}), 400

        # 重新运行 7_analysis.py 并标记点
        try:
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(file_path, pcr_efficiency_file_path, preset_coverage=parameter_value, coding_redundancy=coding_redundancy, t=t)
            message = f"Expected decoding proportion of encoded strands: {np.interp(parameter_value, np.linspace(0.5, 30, 60), Thm1):.3f}"
            preset_coverage = parameter_value
            expected_proportion = None
        except Exception as e:
            logging.error(f"运行 7_analysis.py 失败: {str(e)}")  # 增加日志记录
            return jsonify({"success": False, "message": f"运行 7_analysis.py 失败: {str(e)}"}), 500

    elif parameter_type == "expected_proportion":
        if parameter_value < 0 or parameter_value > 1:
            return jsonify({"success": False, "message": "参数 Expected decoding proportion of encoded strands 必须在 0 到 1 之间！"}), 400

        # 重新运行 7_analysis.py 并标记点
        try:
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(file_path, pcr_efficiency_file_path, expected_proportion=parameter_value, coding_redundancy=coding_redundancy, t=t)
            x_val = np.interp(parameter_value, Thm1, np.linspace(0.5, 30, 60))
            message = f"Preset coverage depth: {x_val:.2f}"
            preset_coverage = x_val
            expected_proportion = parameter_value
        except Exception as e:
            logging.error(f"运行 7_analysis.py 失败: {str(e)}")  # 增加日志记录
            return jsonify({"success": False, "message": f"运行 7_analysis.py 失败: {str(e)}"}), 500

    else:
        return jsonify({"success": False, "message": "无效的参数类型！"}), 400

    # 保存 mu_t 和 sigma_t 到 session
    session["mu_t"] = mu_t
    session["sigma_t"] = sigma_t

    # 刷新图7-1
    return jsonify({"success": True, "message": message, "preset_coverage": preset_coverage, "expected_proportion": expected_proportion, "image_url": url_for('uploaded_file', filename='图7-1.png') + '?t=' + str(int(time.time()))})



@app.route("/update_coding_redundancy", methods=["POST"])
def update_coding_redundancy():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    coding_redundancy = request.json.get("coding_redundancy")
    t = session.get("t")  # 从 session 获取 t 值

    try:
        coding_redundancy = float(coding_redundancy)
        if coding_redundancy < 0.95 or coding_redundancy > 4:
            return jsonify({"success": False, "message": "参数 Coding redundancy 必须在 0.95 到 4 之间！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 Coding redundancy 必须是数字！"}), 400

    # 重新运行 7_analysis.py 并标记点
    try:
        Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(file_path, session.get("pcr_efficiency_file_path"), coding_redundancy=coding_redundancy, t=t)
        message = f"Expected coverage depth: {x_val:.2f}"
    except Exception as e:
        logging.error(f"运行 7_analysis.py 失败: {str(e)}")  # 增加日志记录
        return jsonify({"success": False, "message": f"运行 7_analysis.py 失败: {str(e)}"}), 500

    # 保存 mu_t 和 sigma_t 到 session
    session["mu_t"] = mu_t
    session["sigma_t"] = sigma_t

    # 刷新图7-2
    image_url = url_for('uploaded_file', filename='图7-2.png') + '?t=' + str(int(time.time()))
    return jsonify({"success": True, "message": message, "x_val": x_val, "image_url": image_url})


@app.route("/update_bounds_parameters", methods=["POST"])
def update_bounds_parameters():
    file_path = session.get("file_path")
    pcr_efficiency_file_path = session.get("pcr_efficiency_file_path")
    if not file_path or not pcr_efficiency_file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    R_noisy = request.json.get("R_noisy")
    a = request.json.get("a")

    try:
        R_noisy = float(R_noisy)
        if R_noisy < 1.1:
            return jsonify({"success": False, "message": "参数 R 的最小值为 1.1！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 R 必须是数字！"}), 400

    try:
        a = int(a)
        if a < 2:
            return jsonify({"success": False, "message": "参数 a 的最小值为 2！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 a 必须是整数！"}), 400

    # 获取 mu_t 和 sigma_t 从 session
    mu_t = session.get("mu_t")
    sigma_t = session.get("sigma_t")

    if mu_t is None or sigma_t is None:
        return jsonify({"success": False, "message": "缺少必要的参数 mu_t 或 sigma_t！"}), 400

    # 删除旧的图片文件
    for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
        os.remove(old_image)

    # 保存新的 R_noisy, a 值到 session
    session["R_noisy"] = R_noisy
    session["a"] = a

    # 运行 8_bounds.py 计算上下界
    try:
        result = subprocess.run(
            ["python", "8_bounds.py", file_path, str(session.get("R_noisy")), str(session.get("c")), str(session.get("a")), str(mu_t), str(sigma_t)],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
            encoding='utf-8'
        )
        lines = result.stdout.strip().split('\n')
        lower_bound, center_bound, upper_bound = map(float, lines[:3])
    except subprocess.CalledProcessError as e:
        logging.error(f"运行 8_bounds.py 失败: {e.stderr}")
        return jsonify({"success": False, "message": f"运行 8_bounds.py 失败: {e.stderr}"}), 500

    # 保存上下界到 session
    session["lower_bound"] = lower_bound
    session["upper_bound"] = upper_bound
    session["center_bound"] = center_bound

    return jsonify(
        {"success": True, "message": "Update Successfully！", "lower_bound": lower_bound, "center_bound": center_bound,
         "upper_bound": upper_bound})

@app.route("/update_figure1", methods=["POST"])
def update_figure1():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    choice = request.json.get("choice")
    value = request.json.get("value")
    t = session.get("t")  # 从 session 获取 t 值

    print(f"Fig. 1 received t value: {t}")  # 打印 t 值

    try:
        value = float(value)
    except ValueError:
        return jsonify({"success": False, "message": "参数必须是数字！"}), 400

    if choice == "preset_coverage":
        if value < 0.5 or value > 30:
            return jsonify({"success": False, "message": "参数 Preset coverage depth 必须在 0.5 到 30 之间！"}), 400
        if value % 0.5 != 0:
            return jsonify({"success": False, "message": "参数 Preset coverage depth 必须是 0.5 的倍数！"}), 400

        # 重新运行 7_analysis.py 并标记点
        try:
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(file_path, session.get("pcr_efficiency_file_path"), preset_coverage=value, coding_redundancy=None, t=t)
            message = f"Expected decoding proportion of encoded strands: {np.interp(value, np.linspace(0.5, 30, 60), Thm1):.3f}"
            preset_coverage = value
            expected_proportion = None
        except Exception as e:
            logging.error(f"运行 7_analysis.py 失败: {str(e)}")
            return jsonify({"success": False, "message": f"运行 7_analysis.py 失败: {str(e)}"}), 500

    elif choice == "expected_proportion":
        if value < 0 or value > 1:
            return jsonify({"success": False, "message": "参数 Expected decoding proportion of encoded strands 必须在 0 到 1 之间！"}), 400

        # 重新运行 7_analysis.py 并标记点
        try:
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(file_path, session.get("pcr_efficiency_file_path"), expected_proportion=value, coding_redundancy=None, t=t)
            x_val = np.interp(value, Thm1, np.linspace(0.5, 30, 60))
            message = f"Preset coverage depth: {x_val:.2f}"
            preset_coverage = x_val
            expected_proportion = value
        except Exception as e:
            logging.error(f"运行 7_analysis.py 失败: {str(e)}")
            return jsonify({"success": False, "message": f"运行 7_analysis.py 失败: {str(e)}"}), 500

    else:
        return jsonify({"success": False, "message": "无效的参数类型！"}), 400

    # 保存 mu_t 和 sigma_t 到 session
    session["mu_t"] = mu_t
    session["sigma_t"] = sigma_t

    # 刷新图7-1
    return jsonify({"success": True, "message": message, "preset_coverage": preset_coverage, "expected_proportion": expected_proportion, "image_url": url_for('uploaded_file', filename='图7-1.png') + '?t=' + str(int(time.time()))})


@app.route("/update_figure2", methods=["POST"])
def update_figure2():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    coding_redundancy = request.json.get("coding_redundancy")
    t = session.get("t")  # 从 session 获取 t 值

    print(f"Fig. 2 received t value: {t}")  # 打印 t 值

    try:
        coding_redundancy = float(coding_redundancy)
        if coding_redundancy < 0.95 or coding_redundancy > 4:
            return jsonify({"success": False, "message": "参数 Coding redundancy 必须在 0.95 到 4 之间！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 Coding redundancy 必须是数字！"}), 400

    # 重新运行 7_analysis.py 并标记点
    try:
        Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(file_path, session.get("pcr_efficiency_file_path"), coding_redundancy=coding_redundancy, t=t)
        message = f"Expected coverage depth: {x_val:.2f}"
    except Exception as e:
        logging.error(f"运行 7_analysis.py 失败: {str(e)}")
        return jsonify({"success": False, "message": f"运行 7_analysis.py 失败: {str(e)}"}), 500

    # 保存 mu_t 和 sigma_t 到 session
    session["mu_t"] = mu_t
    session["sigma_t"] = sigma_t

    # 刷新图7-2
    image_url = url_for('uploaded_file', filename='图7-2.png') + '?t=' + str(int(time.time()))
    return jsonify({"success": True, "message": message, "x_val": x_val, "image_url": image_url})


@app.route("/update_noisy_channel", methods=["POST"])
def update_noisy_channel():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "未上传文件！"}), 400

    R_noisy = request.json.get("R_noisy")
    a = request.json.get("a")
    t = request.json.get("t_simulate")  # 获取 t 值

    print(f"Fig. 2 received t value: {t}")  # 打印 t 值

    try:
        R_noisy = float(R_noisy)
        if R_noisy < 1.1:
            return jsonify({"success": False, "message": "参数 R 的最小值为 1.1！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 R 必须是数字！"}), 400

    try:
        a = int(a)
        if a < 2:
            return jsonify({"success": False, "message": "参数 a 的最小值为 2！"}), 400
    except ValueError:
        return jsonify({"success": False, "message": "参数 a 必须是整数！"}), 400

    # 获取 mu_t 和 sigma_t 从 session
    mu_t = session.get("mu_t")
    sigma_t = session.get("sigma_t")

    if mu_t is None or sigma_t is None:
        return jsonify({"success": False, "message": "缺少必要的参数 mu_t 或 sigma_t！"}), 400

    # 删除旧的图片文件
    for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
        os.remove(old_image)

    # 保存新的 R_noisy, a 值到 session
    session["R_noisy"] = R_noisy
    session["a"] = a

    # 运行 8_bounds.py 计算上下界
    try:
        result = subprocess.run(
            ["python", "8_bounds.py", file_path, str(session.get("R_noisy")), str(session.get("c")), str(session.get("a")), str(mu_t), str(sigma_t)],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
            encoding='utf-8'
        )
        lines = result.stdout.strip().split('\n')
        lower_bound, center_bound, upper_bound = map(float, lines[:3])
    except subprocess.CalledProcessError as e:
        logging.error(f"运行 8_bounds.py 失败: {e.stderr}")
        return jsonify({"success": False, "message": f"运行 8_bounds.py 失败: {e.stderr}"}), 500

    # 保存上下界到 session
    session["lower_bound"] = lower_bound
    session["upper_bound"] = upper_bound
    session["center_bound"] = center_bound

    return jsonify(
        {"success": True, "message": "Update Successfully！", "lower_bound": lower_bound, "center_bound": center_bound,
         "upper_bound": upper_bound})

def worker_update_preset_coverage(task_id, file_path, R, preset_coverage, expected_completeness, mark_option):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图4-1.png")):
            os.remove(old_image)

        set_task(task_id, progress=35, message="Running 4_population dist.py ...")
        result = subprocess.run(
            [
                "python",
                "4_population dist.py",
                file_path,
                str(R),
                str(preset_coverage),
                str(expected_completeness),
                mark_option
            ],
            check=True,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        set_task(task_id, progress=85, message="Parsing outputs...")
        lines = parse_nonempty_lines(result.stdout)
        coordinate_info_4_1 = lines[0] if len(lines) > 0 else ""
        coordinate_info_4_2 = lines[1] if len(lines) > 1 else ""

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": "Update successfully!",
                "coordinate_info_4_1": coordinate_info_4_1,
                "coordinate_info_4_2": coordinate_info_4_2
            }
        )
    except subprocess.CalledProcessError as e:
        set_task(task_id, state="error", progress=100, message=f"fail to run 4_population dist.py: {e.stderr}")


def worker_update_R(task_id, file_path, R, preset_coverage, expected_completeness, mark_option):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图4-2.png")):
            os.remove(old_image)

        set_task(task_id, progress=35, message="Running 4_population dist.py ...")
        result = subprocess.run(
            [
                "python",
                "4_population dist.py",
                file_path,
                str(R),
                str(preset_coverage),
                str(expected_completeness),
                mark_option
            ],
            check=True,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        set_task(task_id, progress=85, message="Parsing outputs...")
        lines = parse_nonempty_lines(result.stdout)
        coordinate_info_4_2 = lines[1] if len(lines) > 1 else ""

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": "Update successfully!",
                "coordinate_info_4_2": coordinate_info_4_2
            }
        )
    except subprocess.CalledProcessError as e:
        set_task(task_id, state="error", progress=100, message=f"fail to run 4_population dist.py: {e.stderr}")


def worker_calculate(task_id, file_path, R_noisy, c, a):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
            os.remove(old_image)

        set_task(task_id, progress=45, message="Running 5_bounds.py ...")
        result = subprocess.run(
            ["python", "5_bounds.py", file_path, str(R_noisy), str(c), str(a)],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
            encoding="utf-8"
        )

        set_task(task_id, progress=85, message="Parsing outputs...")
        lines = parse_nonempty_lines(result.stdout)
        lower_bound, center_bound, upper_bound = map(float, lines[:3])

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": "Update successfully!",
                "lower_bound": lower_bound,
                "center_bound": center_bound,
                "upper_bound": upper_bound
            }
        )
    except subprocess.CalledProcessError as e:
        set_task(task_id, state="error", progress=100, message=f"fail to run 5_bounds.py: {e.stderr}")


def worker_update_parameter(task_id, file_path, pcr_efficiency_file_path, parameter_type, parameter_value, coding_redundancy, t):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        set_task(task_id, progress=40, message="Running 7_analysis.py ...")

        if parameter_type == "preset_coverage":
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(
                file_path,
                pcr_efficiency_file_path,
                preset_coverage=parameter_value,
                coding_redundancy=coding_redundancy,
                t=t
            )
            message = f"Expected decoding proportion of encoded strands: {np.interp(parameter_value, np.linspace(0.5, 30, 60), Thm1):.3f}"
            preset_coverage = parameter_value
            expected_proportion = None
        elif parameter_type == "expected_proportion":
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(
                file_path,
                pcr_efficiency_file_path,
                expected_proportion=parameter_value,
                coding_redundancy=coding_redundancy,
                t=t
            )
            x_val = np.interp(parameter_value, Thm1, np.linspace(0.5, 30, 60))
            message = f"Preset coverage depth: {x_val:.2f}"
            preset_coverage = x_val
            expected_proportion = parameter_value
        else:
            raise ValueError("Invalid parameter type")

        set_task(task_id, progress=85, message="Generating figure...")
        image_url = f"/uploads/图7-1.png?t={int(time.time())}"

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": message,
                "preset_coverage": preset_coverage,
                "expected_proportion": expected_proportion,
                "image_url": image_url
            }
        )
    except Exception as e:
        logging.error(f"worker_update_parameter error: {str(e)}")
        set_task(task_id, state="error", progress=100, message=str(e))


def worker_update_coding_redundancy(task_id, file_path, pcr_efficiency_file_path, coding_redundancy, t):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        set_task(task_id, progress=40, message="Running 7_analysis.py ...")

        Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(
            file_path,
            pcr_efficiency_file_path,
            coding_redundancy=coding_redundancy,
            t=t
        )
        message = f"Expected coverage depth: {x_val:.2f}"

        set_task(task_id, progress=85, message="Generating figure...")
        image_url = f"/uploads/图7-2.png?t={int(time.time())}"

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": message,
                "x_val": x_val,
                "image_url": image_url
            }
        )
    except Exception as e:
        logging.error(f"worker_update_coding_redundancy error: {str(e)}")
        set_task(task_id, state="error", progress=100, message=str(e))


def worker_update_bounds_parameters(task_id, file_path, R_noisy, c, a, mu_t, sigma_t):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        for old_image in glob.glob(os.path.join(UPLOAD_FOLDER, "图*.png")):
            os.remove(old_image)

        set_task(task_id, progress=45, message="Running 8_bounds.py ...")
        result = subprocess.run(
            [
                "python",
                "8_bounds.py",
                file_path,
                str(R_noisy),
                str(c),
                str(a),
                str(mu_t),
                str(sigma_t)
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
            encoding="utf-8"
        )

        set_task(task_id, progress=85, message="Parsing outputs...")
        lines = parse_nonempty_lines(result.stdout)
        lower_bound, center_bound, upper_bound = map(float, lines[:3])

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": "Update successfully!",
                "lower_bound": lower_bound,
                "center_bound": center_bound,
                "upper_bound": upper_bound
            }
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"worker_update_bounds_parameters error: {e.stderr}")
        set_task(task_id, state="error", progress=100, message=e.stderr)


def worker_update_figure1(task_id, file_path, pcr_efficiency_file_path, choice, value, t):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        set_task(task_id, progress=40, message="Running 7_analysis.py ...")

        if choice == "preset_coverage":
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(
                file_path,
                pcr_efficiency_file_path,
                preset_coverage=value,
                coding_redundancy=None,
                t=t
            )
            message = f"Expected decoding proportion of encoded strands: {np.interp(value, np.linspace(0.5, 30, 60), Thm1):.3f}"
            preset_coverage = value
            expected_proportion = None
        elif choice == "expected_proportion":
            Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(
                file_path,
                pcr_efficiency_file_path,
                expected_proportion=value,
                coding_redundancy=None,
                t=t
            )
            x_val = np.interp(value, Thm1, np.linspace(0.5, 30, 60))
            message = f"Preset coverage depth: {x_val:.2f}"
            preset_coverage = x_val
            expected_proportion = value
        else:
            raise ValueError("Invalid parameter type")

        set_task(task_id, progress=85, message="Generating figure...")
        image_url = f"/uploads/图7-1.png?t={int(time.time())}"

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": message,
                "preset_coverage": preset_coverage,
                "expected_proportion": expected_proportion,
                "image_url": image_url
            }
        )
    except Exception as e:
        logging.error(f"worker_update_figure1 error: {str(e)}")
        set_task(task_id, state="error", progress=100, message=str(e))


def worker_update_figure2(task_id, file_path, pcr_efficiency_file_path, coding_redundancy, t):
    try:
        set_task(task_id, progress=10, message="Preparing parameters...")
        set_task(task_id, progress=40, message="Running 7_analysis.py ...")

        Thm1, Thm2, mu_t, sigma_t, x_val = module.analyze_and_plot(
            file_path,
            pcr_efficiency_file_path,
            coding_redundancy=coding_redundancy,
            t=t
        )
        message = f"Expected coverage depth: {x_val:.2f}"

        set_task(task_id, progress=85, message="Generating figure...")
        image_url = f"/uploads/图7-2.png?t={int(time.time())}"

        set_task(
            task_id,
            state="done",
            progress=100,
            message="Completed",
            result={
                "message": message,
                "x_val": x_val,
                "image_url": image_url
            }
        )
    except Exception as e:
        logging.error(f"worker_update_figure2 error: {str(e)}")
        set_task(task_id, state="error", progress=100, message=str(e))

@app.route("/start_update_preset_coverage", methods=["POST"])
def start_update_preset_coverage():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    preset_coverage = request.form.get("preset_coverage", session.get("preset_coverage", 5))
    expected_completeness = request.form.get("expected_completeness", session.get("expected_completeness", 0.5))
    mark_option = request.form.get("mark_option", session.get("mark_option", "preset_coverage"))

    try:
        preset_coverage = float(preset_coverage)
        expected_completeness = float(expected_completeness)
    except ValueError:
        return jsonify({"success": False, "message": "Parameters must be numeric!"}), 400

    session["preset_coverage"] = preset_coverage
    session["expected_completeness"] = expected_completeness
    session["mark_option"] = mark_option

    task_id = create_task()
    threading.Thread(
        target=worker_update_preset_coverage,
        args=(task_id, file_path, session.get("R"), preset_coverage, expected_completeness, mark_option),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_update_R", methods=["POST"])
def start_update_R():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    R = request.form.get("R", session.get("R", 2))
    try:
        R = float(R)
    except ValueError:
        return jsonify({"success": False, "message": "R must be numeric!"}), 400

    session["R"] = R

    task_id = create_task()
    threading.Thread(
        target=worker_update_R,
        args=(
            task_id,
            file_path,
            R,
            session.get("preset_coverage", 5),
            session.get("expected_completeness", 0.5),
            session.get("mark_option", "preset_coverage")
        ),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_calculate", methods=["POST"])
def start_calculate():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    R_noisy = request.form.get("R_noisy", session.get("R_noisy", 2))
    a = request.form.get("a", session.get("a", 2))
    c = request.form.get("c", session.get("c", 0.01))

    try:
        R_noisy = float(R_noisy)
        a = int(a)
        c = float(c)
    except ValueError:
        return jsonify({"success": False, "message": "Parameters must be numeric!"}), 400

    session["R_noisy"] = R_noisy
    session["a"] = a
    session["c"] = c

    task_id = create_task()
    threading.Thread(
        target=worker_calculate,
        args=(task_id, file_path, R_noisy, c, a),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_update_parameter", methods=["POST"])
def start_update_parameter():
    file_path = session.get("file_path")
    pcr_efficiency_file_path = session.get("pcr_efficiency_file_path")
    if not file_path or not pcr_efficiency_file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    parameter_type = request.json.get("parameter_type")
    parameter_value = request.json.get("parameter_value")
    coding_redundancy = request.json.get("coding_redundancy")

    try:
        parameter_value = float(parameter_value)
    except ValueError:
        return jsonify({"success": False, "message": "Parameters must be numeric!"}), 400

    task_id = create_task()
    threading.Thread(
        target=worker_update_parameter,
        args=(task_id, file_path, pcr_efficiency_file_path, parameter_type, parameter_value, coding_redundancy, session.get("t")),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_update_coding_redundancy", methods=["POST"])
def start_update_coding_redundancy():
    file_path = session.get("file_path")
    pcr_efficiency_file_path = session.get("pcr_efficiency_file_path")
    if not file_path or not pcr_efficiency_file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    coding_redundancy = request.json.get("coding_redundancy")
    try:
        coding_redundancy = float(coding_redundancy)
    except ValueError:
        return jsonify({"success": False, "message": "Coding redundancy must be numeric!"}), 400

    task_id = create_task()
    threading.Thread(
        target=worker_update_coding_redundancy,
        args=(task_id, file_path, pcr_efficiency_file_path, coding_redundancy, session.get("t")),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_update_bounds_parameters", methods=["POST"])
def start_update_bounds_parameters():
    file_path = session.get("file_path")
    if not file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    R_noisy = request.json.get("R_noisy")
    a = request.json.get("a")

    try:
        R_noisy = float(R_noisy)
        a = int(a)
    except ValueError:
        return jsonify({"success": False, "message": "Parameters must be numeric!"}), 400

    session["R_noisy"] = R_noisy
    session["a"] = a

    mu_t = session.get("mu_t")
    sigma_t = session.get("sigma_t")
    c = session.get("c", 0.01)

    if mu_t is None or sigma_t is None:
        return jsonify({"success": False, "message": "Missing mu_t or sigma_t!"}), 400

    task_id = create_task()
    threading.Thread(
        target=worker_update_bounds_parameters,
        args=(task_id, file_path, R_noisy, c, a, mu_t, sigma_t),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_update_figure1", methods=["POST"])
def start_update_figure1():
    file_path = session.get("file_path")
    pcr_efficiency_file_path = session.get("pcr_efficiency_file_path")
    if not file_path or not pcr_efficiency_file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    choice = request.json.get("choice")
    value = request.json.get("value")

    try:
        value = float(value)
    except ValueError:
        return jsonify({"success": False, "message": "Parameters must be numeric!"}), 400

    task_id = create_task()
    threading.Thread(
        target=worker_update_figure1,
        args=(task_id, file_path, pcr_efficiency_file_path, choice, value, session.get("t")),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/start_update_figure2", methods=["POST"])
def start_update_figure2():
    file_path = session.get("file_path")
    pcr_efficiency_file_path = session.get("pcr_efficiency_file_path")
    if not file_path or not pcr_efficiency_file_path:
        return jsonify({"success": False, "message": "No uploaded file!"}), 400

    coding_redundancy = request.json.get("coding_redundancy")
    try:
        coding_redundancy = float(coding_redundancy)
    except ValueError:
        return jsonify({"success": False, "message": "Coding redundancy must be numeric!"}), 400

    task_id = create_task()
    threading.Thread(
        target=worker_update_figure2,
        args=(task_id, file_path, pcr_efficiency_file_path, coding_redundancy, session.get("t")),
        daemon=True
    ).start()

    return jsonify({"success": True, "task_id": task_id})

if __name__ == "__main__":
    app.run(debug=True)
