import tkinter as tk
from tkinter import filedialog, messagebox
import threading

#استخدمت تشات جي بي تي عشان يساعدني في هل جزئية كونه ما عندي هل خبرة في اني اسوي واجهة للمستخدم
def choose_file_and_ratios():
    root = tk.Tk()
    root.title("اختيار ملف البيانات")
    root.geometry("400x400")
    root.config(bg="#f4f4f9")

    file_path = ""
    train_ratio = 0.70
    validation_ratio = 0.15

    def load_file():
        nonlocal file_path
        file_path = filedialog.askopenfilename(title="اختر ملف البيانات", filetypes=[("CSV files", "*.csv")])
        if file_path:
            file_label.config(text=f"تم اختيار الملف: {file_path.split('/')[-1]}")
            load_button.config(state=tk.NORMAL)

    def update_test_ratio(*args):
        try:
            t_ratio = float(train_ratio_entry.get())
            v_ratio = float(validation_ratio_entry.get())
            if t_ratio + v_ratio <= 1.0:
                test_ratio = 1.0 - (t_ratio + v_ratio)
                test_ratio_label.config(text=f"نسبة الاختبار: {test_ratio * 100:.1f}%")
            else:
                test_ratio_label.config(text="❌ النسب غير صحيحة")
        except ValueError:
            test_ratio_label.config(text="❌ أدخل أرقامًا صحيحة")

    def reset_fields():
        train_ratio_entry.delete(0, tk.END)
        train_ratio_entry.insert(0, "0.70")
        validation_ratio_entry.delete(0, tk.END)
        validation_ratio_entry.insert(0, "0.15")
        test_ratio_label.config(text="نسبة الاختبار: 15%")
        file_label.config(text="لم يتم اختيار ملف بعد")
        load_button.config(state=tk.NORMAL)

    def submit_ratios():
        nonlocal train_ratio, validation_ratio
        try:
            train_ratio = float(train_ratio_entry.get())
            validation_ratio = float(validation_ratio_entry.get())
            if train_ratio + validation_ratio < 1.0:
                test_ratio = 1.0 - (train_ratio + validation_ratio)
                messagebox.showinfo("نجاح", f"✅ تم تقسيم البيانات:\n - تدريب: {round(train_ratio * 100,2)}% \n - تحقق: {round(validation_ratio * 100,2)}% \n - اختبار: {round(test_ratio * 100,2)}%")
                root.quit()
            else:
                raise ValueError("النسب غير صحيحة، تأكد من أن مجموع النسب لا يتجاوز 100%")
        except ValueError as e:
            messagebox.showerror("خطأ", str(e))

    tk.Label(root, text="إعدادات تحميل البيانات", font=("Arial", 16, "bold"), bg="#f4f4f9", fg="#333").pack(pady=10)

    file_label = tk.Label(root, text="لم يتم اختيار ملف بعد", width=40, anchor='w', bg="#f4f4f9", font=("Arial", 12), fg="#555")
    file_label.pack(padx=20, pady=5)

    load_button = tk.Button(root, text="اختيار ملف", command=load_file, font=("Arial", 12), bg="#4CAF50", fg="white", width=20, height=2)
    load_button.pack(pady=10)

    ratio_frame = tk.Frame(root, bg="#f4f4f9")
    ratio_frame.pack(pady=10)

    tk.Label(ratio_frame, text="نسبة التدريب (مثلاً 0.70)", font=("Arial", 12), bg="#f4f4f9").grid(row=0, column=0, padx=10, pady=5)
    train_ratio_entry = tk.Entry(ratio_frame, font=("Arial", 12))
    train_ratio_entry.insert(0, "0.70")
    train_ratio_entry.grid(row=0, column=1, padx=10, pady=5)
    train_ratio_entry.bind("<KeyRelease>", update_test_ratio)

    tk.Label(ratio_frame, text="نسبة التحقق (مثلاً 0.15)", font=("Arial", 12), bg="#f4f4f9").grid(row=1, column=0, padx=10, pady=5)
    validation_ratio_entry = tk.Entry(ratio_frame, font=("Arial", 12))
    validation_ratio_entry.insert(0, "0.15")
    validation_ratio_entry.grid(row=1, column=1, padx=10, pady=5)
    validation_ratio_entry.bind("<KeyRelease>", update_test_ratio)

    test_ratio_label = tk.Label(root, text="نسبة الاختبار: 15%", font=("Arial", 12, "bold"), bg="#f4f4f9", fg="#d32f2f")
    test_ratio_label.pack(pady=10)

    button_frame = tk.Frame(root, bg="#f4f4f9")
    button_frame.pack(pady=10)

    submit_button = tk.Button(button_frame, text="إرسال", command=lambda: threading.Thread(target=submit_ratios).start(), font=("Arial", 14, "bold"), bg="#FF5722", fg="white", width=15, height=2)
    submit_button.grid(row=0, column=0, padx=10, pady=5)

    reset_button = tk.Button(button_frame, text="إعادة ضبط", command=reset_fields, font=("Arial", 12), bg="#2196F3", fg="white", width=15, height=2)
    reset_button.grid(row=0, column=1, padx=10, pady=5)

    root.mainloop()

    return file_path, train_ratio, validation_ratio
