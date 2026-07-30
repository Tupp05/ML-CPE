import pandas as pd #ใช้อ่านและจัดการข้อมูลผู้ป่วย
from sklearn.model_selection import train_test_split #ใช้แบ่งข้อมูลไป Train และ Test
from sklearn.compose import ColumnTransformer #ใช้บอกว่าคอลัมน์แต่ละประเภทต้องแปลงเป็นอะไร
from sklearn.preprocessing import OneHotEncoder #ใช้แปลงข้ัอมูลให้เป็นตัวเลขหรือBoolean
from sklearn.pipeline import Pipeline #ใช้เชื่อมโยงขั้นตอน
from sklearn.tree import DecisionTreeClassifier #โมเดลที่ใช้ตัดสินใจ
from sklearn.metrics import accuracy_score #เช็คความแม่นยำของโมเดล
from sklearn.metrics import classification_report #เช็คความแม่นยำของโมเดลแต่ละเอียดกว่า

df=pd.read_csv(r"C:\Users\tuppo\OneDrive\เดสก์ท็อป\ML Lab\silent_sting_triage_data.csv") #อ่านไฟล์

print("---------- Missing Value ----------")
print(df.isnull().sum()) #ตรวจสอบข้อมูลที่หาย หรือ เป็นช่องว่าง
print()
df["Local_Swelling"] = df["Local_Swelling"].fillna("None") #เติมข้อมูลที่เป็นช่องว่างให้เป็น None
print("------- Fixed Missing Value -------")
print(df.isnull().sum()) #ตรวจสอบข้อมูลที่หายหลังแก้ไข
print()

y = df["Bite_Source_Target"] #สิ้งที่จะทำนาย
x = df.drop(columns=["Bite_Source_Target", "Patient_ID"]) #ข้อมูลที่ใช้ทำนาย(ข้อมูลทั้งหมดยกเว้น Bite_Source_Target และ Patient_ID)

#แยกประเภทข้อมูลที่เป็นประเภทและตัวเลข
categorical_cols = ["Gender","Local_Swelling"]
numeric_cols = ["Age", "Time_Since_Bite_Min", "Heart_Rate_BPM", "Blood_Pressure_Systolic", "Muscle_Paralysis_Present", "Blood_Coagulation_Failure"]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=1) #แยกข้อมูลเพื่อเอาไปTrainและTestแบบสุ่ม Train 80% และ Test 20%

#แปลงข้อมูลจากคอลลั่ม categorical_cols ให้เป็นตัวเลข ถ้ามีข้อมูลที่ไม่รู้จักให้ใส่ ignore ถ้าไม่ใช่คอลลั่มนี้ให้ผ่านได้เลย
preprocessor = ColumnTransformer(transformers = [("category", OneHotEncoder(handle_unknown = "ignore"), categorical_cols)], remainder="passthrough")

#เชื่อมขั้นตอนเอาผลลัพพ์จาก preprocessor ไปตัดสินใจต่อใน DecisionTreeClassifier
model = Pipeline([("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier(random_state=1))])

model.fit(x_train, y_train) #เทรนโมเดล

print("------------------------- Prediction -------------------------")
y_predicted = model.predict(x_test) #ให้โมเดลpredictข้อมูลtest 
print(y_predicted) #แสดงผลลัพธ์การ predict
print()

accuracy = accuracy_score(y_test, y_predicted) #ตรวจสอบความถูกต้องของข้อมูลที่predictกับผลลัพธ์จริง
print("Accuracy : ", accuracy, "or", accuracy*100,"%") #แสดงความแม่นยำ
print()

print("------------------- Report of Prediction -------------------")
print(classification_report(y_test, y_predicted, digits=6)) #แสดงรายละเอียดของความแม่นยำ

#souce : https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset
print("Souce : Emergency Triage: Venomous Bites Dataset (https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset)")