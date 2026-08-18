"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import os
import re
import sys
import subprocess


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def configure_console() -> None:
    """Keep the checker usable on Windows consoles with a cp1252 default."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        print(f"  ✅ {path} — keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    """Run pytest and return (passed, total)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "-q"],
            capture_output=True, text=True, timeout=120,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        # Pytest may wrap the summary in separator characters, e.g.
        # "===================== 37 passed in 64s =====================".
        matches = re.findall(
            r"(?P<count>\d+)\s+(?P<status>passed|failed|error|errors|skipped|xfailed|xpassed)",
            output,
            flags=re.IGNORECASE,
        )
        if not matches:
            print("  ⚠️  pytest không trả về dòng tổng kết hợp lệ")
            return 0, 0

        counts = {"passed": 0, "failed": 0, "error": 0, "errors": 0,
                  "skipped": 0, "xfailed": 0, "xpassed": 0}
        for count, status in matches:
            counts[status.lower()] += int(count)
        total = sum(counts.values())
        return counts["passed"], total
    except Exception as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0


def validate():
    os.chdir(ROOT_DIR)
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/m5_enrichment.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    for f in ["analysis/failure_analysis.md", "analysis/group_report.md"]:
        if not check_file(f):
            errors += 1

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections = [f for f in os.listdir(ref_dir) if f.startswith("reflection_") and f.endswith(".md")]
    if reflections:
        for r in reflections:
            print(f"  ✅ {ref_dir}/{r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân trong {ref_dir}/")
        errors += 1

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")
        errors += 1

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if pct >= 80 else '⚠️'} {passed}/{total} tests passed ({pct:.0f}%)")
        if passed < total:
            errors += 1
    else:
        print("  ⚠️  Không chạy được tests")
        errors += 1

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)
    return errors


if __name__ == "__main__":
    configure_console()
    raise SystemExit(validate())
