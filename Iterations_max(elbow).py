import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import janitor
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

filepath = 'C:\\Users\\pasante.coop\\OneDrive - DIFARE S.A\\Documentos\\Python\\Proyectos\\reporte total cartera 31082025.xlsx'
df= pd.read_excel(filepath)
df= pd.DataFrame(df)
df_clean= df.clean_names()

y= df_clean['calificacion']
x= df_clean[['saldo_por_vencer','saldo_vencido','saldo_no_devenga','valor_garantia','provision','dias_mora','nro_cuotas_transcurrido','edad_cliente','estado_civil','totalinteres','mora','saldo','tasa_interes','monto_original','valorcuota','interes','interesvencido','interesganado','interesporcobrar','totalinteres']]
x= pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

max_iters = range(1600, 3001, 100)  # desde 1600 hasta 3000 iteraciones en pasos de 100
accuracies = []

for mi in max_iters:
    model = LogisticRegression(max_iter=mi)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure(figsize=(10,6))
plt.plot(max_iters, accuracies, marker='o')
plt.xlabel('max_iter')
plt.ylabel('Accuracy')
plt.title('Accuracy vs max_iter en Regresión Logística')
plt.grid(True)
plt.show()