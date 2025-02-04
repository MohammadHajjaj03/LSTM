import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
import Choose_file as CF

#هون احنا بنحدد قيم ثابتة للقيم العشوائية الي بتدخل في تحديد الاوزان في بدايةكل Run
#فعملنا هيك عشان نظمن انه النتائج النهائية ما بتتغير كل ما نعمل Run
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

#هون احنا اعطينا مدخلات للمتغيرات الي عنا مسار الملف او نسبة التدريب او نسبة التحقق
#عبر انها استدعينا فكنشن من المكتبة الي سويناها
file_path, train_ratio, validation_ratio = CF.choose_file_and_ratios()

#شرط بسيط عشان اتاكد اذا كلشي عندي ماشي صح ولا لا
if file_path:

    #مبدئيا هون ببساطة احنا قاعدين بنستدعي الداتا الي عنا وبنفس الوقت بنسوي الها شوية تعديلات ونوع من البري بروسسنق
    Data_set = pd.read_csv(file_path)
    print("تم تحميل الملف بنجاح:", file_path)

    Data_set['Date']=pd.to_datetime(Data_set['Date'])#هون احنا خلينا نوع عمود التاريخ يتحول من object ل date_time عشان نقدر نستفيد منه بالمودل
    Data_set.set_index('Date',inplace=True)#هون ببساطة يعني احنا خلينا ترتيب الداتا تبعا للتاريخ مو لارقامها في الاكسل شيت
    Target_data=Data_set['Close']



    #هون قاعدين بنقسم الداتا لكلشي بنحتاجه
    train_data = int(len(Target_data) * train_ratio)
    validation_data = int(len(Target_data) * (train_ratio + validation_ratio))
    train= Target_data[:train_data]
    validation=Target_data[train_data:validation_data]
    test=Target_data[validation_data:]
    test_ratio = round(1.00 - (train_ratio + validation_ratio), 2)
    #رسم البيانات وتوضيحها على حسب الفترات
    plt.figure(figsize=(12, 6))
    plt.plot(Target_data.index[:train_data], Target_data.values[:train_data], label=f'Training Data ({train_ratio*100}%)', color='skyblue')
    plt.plot(Target_data.index[train_data:validation_data], Target_data.values[train_data:validation_data], label=f'Validation Data ({validation_ratio}%)', color='salmon')
    plt.plot(Target_data.index[validation_data:], Target_data.values[validation_data:], label=f'Testing Data ({test_ratio}%)', color='lightgreen')
    plt.title('Stock Price with Training, Validation, and Testing Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='best')
    plt.show()

    #هون اعطينا متغير السكيلر خليناه يوخذ فيت او انه يتدرب على تحويل الداتا train ل قيم ما بين 0 و 1
    scaler = MinMaxScaler()
    scaler.fit(train.values.reshape(-1,1))
    #قاعدين بنغير كل نوع من الداتا لنفس شكل الداتا تبعت ال train
    scaled_train=scaler.transform(train.values.reshape(-1,1))
    scaled_validation=scaler.transform(validation.values.reshape(-1,1))
    scaled_test=scaler.transform(test.values.reshape(-1,1))


    #هاظ الفنكشن شغله انه يعني
    #windows split ببساطة يقسم الداتا بشكل او باخر
    def Data_into_x_y(num_of_steps,Data):
        x,y=[],[]
        for i in range(len(Data)-num_of_steps):
            x.append(Data[i:i+num_of_steps])
            y.append(Data[i+num_of_steps])
        return np.array(x),np.array(y)

    num_of_steps=10
    #استدعينا الفنكشن واعطيناه مدخلات وهو اعطانا مخرجات
    train_x,train_y=Data_into_x_y(num_of_steps,scaled_train)
    validation_x,validation_y=Data_into_x_y(num_of_steps,scaled_validation)
    test_x,test_y=Data_into_x_y(num_of_steps,scaled_test)
    #هون بنبني ال architecture تبع المودل
    #هون بنضيف ال layers الي بنحتاجها في المودل
    model = Sequential([
        LSTM(64,activation="relu", return_sequences=True ,recurrent_dropout=0.2, input_shape=(num_of_steps, 1)),
        LSTM(32,activation="relu",recurrent_dropout=0.2),
        Dense(25),
        Dropout(0.2),
        Dense(1)
    ])

    #هون بحدد الامور الي رح يستند الها المودل وقت التدريب مثلا الوسس فنكشن او الابتمايزر وغيره
    model.compile(loss=MeanSquaredError(),optimizer=Adam(learning_rate=0.0007),metrics=[RootMeanSquaredError()])
    #هون خليناه يحفظ افضل مودل
    Checkpoint=ModelCheckpoint('model/best_model.keras',save_best_only=True)
    #هون حرفيا بنبدا ندرب المودل بعد ما اعطيناه الداتا والايبوكس وغيرها
    history=model.fit(train_x,train_y,validation_data=(validation_x,validation_y),epochs=50,batch_size=32,callbacks=[Checkpoint])
    #هون بعطينا رسم بياني بورينا الاخطاء في التدريب والاخطاء في التحقق على مدى عمل المودل
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

    best_model = load_model('model/best_model.keras')
    #هون بنعطيه امر انه يتوقع التست داتا
    test_pred=best_model.predict(test_x).flatten()
    #هون بنرجع الداتا لزي ما كانت بالداتا سيت عشان نقدر نقارنها ونشوف وضعنا كيف
    original_pred_values=scaler.inverse_transform(test_pred.reshape(-1,1)).flatten()
    original_test_values=scaler.inverse_transform(test_y.reshape(-1,1)).flatten()
    #هون احنا حطينا الداتا الاساسية والداتا المتوقعة في داتا فريم وحدة عشان نقدر نعرضها
    test_result  = pd.DataFrame(data={'test prediction': original_pred_values, 'actual': original_test_values})
    print(test_result)
    #بنحسب بعض الامور المهمة عشان نشوف مدى دقة المودل
    mse=mean_squared_error(test_y,test_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(test_y,test_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")

    #بنعمل قاعدين رسم بياني يعرض النا الداتا الاصلية والداتا المتوقعة وكم التباين بينهم
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=test_result.index, y=test_result['actual'], label='Actual', color='blue', linewidth=2)
    sns.lineplot(x=test_result.index, y=test_result['test prediction'], label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.title("Actual vs Predicted Stock Prices", fontsize=16, fontweight='bold')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Close Price", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


    #بنرسم توزيع الاخطاء واذا كانت بتتوزع حولين الصفر فغالبا امورنا بالتمام
    #اذا كان في انحياز لليمين بكون بتوقع اعلى واذا لليسار بكون اقل
    plt.figure(figsize=(8, 6))
    sns.histplot(test_result['actual'] - test_result['test prediction'], kde=True, color="skyblue", bins=30)
    plt.title("Error Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Prediction Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.show()


    #بنحفظ المودل
    model.save('apple_stock_predictor.keras')
    #رسم بياني بوضحلي العلاقة بين الاسعار الفعلية والاسعار الي توقعها المودل الخط الاحمر هي النقطة المثالية وكل نقطة بتمثل السعر الي توقعه المودل مقابل السعر الحقيقي
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=test_result['actual'], y=test_result['test prediction'], alpha=0.6, edgecolor=None)
    plt.plot([test_result['actual'].min(), test_result['actual'].max()],
             [test_result['actual'].min(), test_result['actual'].max()],
             color='red', linestyle='--', linewidth=2)
    plt.title("Scatter Plot: Actual vs Predicted Prices", fontsize=16, fontweight='bold')
    plt.xlabel("Actual Prices", fontsize=12)
    plt.ylabel("Predicted Prices", fontsize=12)
    plt.grid(True)
    plt.show()

    #طبعا هون رسم بياني بوضح السعر الفعلي والسعر المتوقع وبرضو معهم المتوسط المتحرك
    #كونه المتوسط المتحرك يساعدنا نشوف الاتجاه العام للسعر بدون تأثير التذبذبات اليومية
    window = 10
    plt.figure(figsize=(12, 6))
    plt.plot(test_result['actual'], label="Actual Prices", color='blue')
    plt.plot(test_result['test prediction'], label="Predicted Prices", color='red', linestyle='--')
    plt.plot(test_result['actual'].rolling(window).mean(), label=f"{window}-Day Rolling Mean", color='green', linestyle='-.')
    plt.legend()
    plt.title("Stock Prices with Rolling Mean")
    plt.show()
else:
    print("❌ لم يتم اختيار أي ملف!")