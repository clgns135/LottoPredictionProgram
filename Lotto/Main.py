import subprocess
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def predict_numbers():
    result_text.delete(1.0, tk.END)
    command = "python Lotto.py"  # Lotto.py 실행 명령어
    result = subprocess.check_output(command, shell=True).decode("utf-8")
    result_text.insert(tk.END, result)

def show_frequency():
    result_text.delete(1.0, tk.END)
    command = "python Frequency.py"  # Frequency.py 실행 명령어
    result = subprocess.check_output(command, shell=True).decode("utf-8")
    result_text.insert(tk.END, result)

def random_numbers():
    result_text.delete(1.0, tk.END)
    command = "python RandomLotto.py"  # RandomLotto.py 실행 명령어
    result = subprocess.check_output(command, shell=True).decode("utf-8")
    result_text.insert(tk.END, result)

def show_calculate():
    result_text.delete(1.0, tk.END)
    command = "python Calculate.py"  # Calculate.py 실행 명령어
    result = subprocess.check_output(command, shell=True).decode("utf-8")
    result_text.insert(tk.END, get_desired_results(result))

def get_desired_results(result):
    lines = result.split("\n")
    desired_results = []
    for line in lines:
        if line.startswith("학습 데이터 정확도:") or line.startswith("테스트 데이터 정확도:") or line.startswith("MSE:") or line.startswith("R-squared:"):
            desired_results.append(line)
    return "\n".join(desired_results)

# GUI 창 생성
window = tk.Tk()
window.title("Lotto Predictor")
window.configure(bg="white")  # 배경색 변경
window.resizable(False, False)  # 확장 및 축소 비활성화

# 백그라운드 이미지 로드
background_image = Image.open("background.jpg")
background_image = background_image.resize((int(window.winfo_screenwidth()/3.1), int(window.winfo_screenheight()/1.1)), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)

# 라벨에 백그라운드 이미지 설정
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# 이미지 로드
image = Image.open("Lotto.jpeg")
image = image.resize((600, 400), Image.LANCZOS)  # 이미지 크기 조정
photo = ImageTk.PhotoImage(image)

# 라벨에 이미지 설정
image_label = tk.Label(window, image=photo)
image_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="nsew")

# 버튼 스타일
button_style = ttk.Style()
button_style.configure(
    "Custom.TButton",
    relief=tk.RAISED,
    font=("맑은 고딕", 16),
    background="#FFD700",  # 버튼 색상 변경 (금색)
    foreground="#000000",  # 버튼 텍스트 색상 변경 (검은색)
    bordercolor="#FFD700",  # 버튼 테두리 색상 변경 (금색)
    width=20,  # 버튼 가로 크기 변경
    height=3,  # 버튼 세로 크기 변경
)

# 버튼 1과 2 배치
button1 = ttk.Button(window, text="다음 숫자 예측", command=predict_numbers, style="Custom.TButton")
button1.grid(row=1, column=0, padx=0, pady=10, sticky="nsew")

button2 = ttk.Button(window, text="빈도수 확인", command=show_frequency, style="Custom.TButton")
button2.grid(row=1, column=1, padx=0, pady=10, sticky="nsew")

# 버튼 3와 4 배치
button3 = ttk.Button(window, text="랜덤 번호 생성", command=random_numbers, style="Custom.TButton")
button3.grid(row=2, column=0, padx=0, pady=10, sticky="nsew")

button4 = ttk.Button(window, text="정확도 확인", command=show_calculate, style="Custom.TButton")
button4.grid(row=2, column=1, padx=0, pady=10, sticky="nsew")

# 결과 출력용 텍스트 박스 생성
result_frame = ttk.Frame(window)
result_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

scrollbar = tk.Scrollbar(result_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

result_text = tk.Text(result_frame, width=40, height=10, yscrollcommand=scrollbar.set)
result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
result_text.configure(font=("맑은 고딕", 12))  # 텍스트 폰트 변경

scrollbar.config(command=result_text.yview)

# Grid 열과 행 비율 조정
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
window.rowconfigure(2, weight=1)
window.rowconfigure(3, weight=1)

window.mainloop()
