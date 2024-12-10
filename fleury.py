import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image, ImageTk
import sys
import subprocess
import warnings

# Função para instalar pacotes faltantes
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Tentar importar Pillow, caso não esteja instalado
try:
    from PIL import Image, ImageTk
except ImportError:
    install_package("Pillow")
    from PIL import Image, ImageTk

class AplicativoAnaliseDados:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Dados Clínicos")
        self.root.geometry("600x600")
        
        # Inicializar DataFrame
        self.df = None
        
        # Botão para importar dados
        self.btn_importar = tk.Button(
            root,
            text="Importar Dados CSV",
            command=self.importar_dados,
            width=30,
            height=2
        )
        self.btn_importar.pack(pady=20)
        
        # Botão para análise estatística
        self.btn_analise = tk.Button(
            root,
            text="Análise Estatística",
            command=self.analise_estatistica,
            width=30,
            height=2,
            state=tk.DISABLED,
        )
        self.btn_analise.pack(pady=10)
        
        # Frame para os botões de modelos
        self.frame_modelos = tk.Frame(root)
        self.frame_modelos.pack(pady=10)
        
        # Botão para modelo Random Forest
        self.btn_random_forest = tk.Button(
            self.frame_modelos,
            text="Random Forest",
            command=lambda: self.executar_modelo("Random Forest"),
            width=20,
            height=2,
            state=tk.DISABLED,
            bg='lightblue'
        )
        self.btn_random_forest.grid(row=0, column=0, padx=10, pady=5)
        
        # Botão para modelo SVM
        self.btn_svm = tk.Button(
            self.frame_modelos,
            text="Support Vector Machine (SVM)",
            command=lambda: self.executar_modelo("SVM"),
            width=25,
            height=2,
            state=tk.DISABLED,
            bg='lightgreen'
        )
        self.btn_svm.grid(row=0, column=1, padx=10, pady=5)
        
        # Botão para modelo KNN
        self.btn_knn = tk.Button(
            self.frame_modelos,
            text="K-Nearest Neighbors (KNN)",
            command=lambda: self.executar_modelo("KNN"),
            width=25,
            height=2,
            state=tk.DISABLED,
            bg='lightyellow'
        )
        self.btn_knn.grid(row=1, column=0, padx=10, pady=5)
        
    def importar_dados(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar arquivo CSV",
            filetypes=(("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*"))
        )
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                messagebox.showinfo(
                    "Sucesso",
                    f"Arquivo carregado com sucesso!\nLinhas: {self.df.shape[0]}, Colunas: {self.df.shape[1]}"
                )
                self.btn_analise.config(state=tk.NORMAL)
                self.btn_random_forest.config(state=tk.NORMAL)
                self.btn_svm.config(state=tk.NORMAL)
                self.btn_knn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao ler o arquivo CSV.\nErro: {e}")
                
    def analise_estatistica(self):
        if self.df is not None:
            # Selecionar colunas numéricas
            colunas_numericas = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if not colunas_numericas:
                messagebox.showwarning("Aviso", "Nenhuma coluna numérica encontrada para análise.")
                return
            
            # Criar pasta para salvar gráficos
            pasta_graficos = "graficos_analise"
            if not os.path.exists(pasta_graficos):
                os.makedirs(pasta_graficos)
            
            for coluna in colunas_numericas:
                plt.figure(figsize=(12, 5))
                
                # Histograma
                plt.subplot(1, 2, 1)
                sns.histplot(self.df[coluna], kde=True, bins=30, color='skyblue')
                plt.title(f'Histograma de {coluna}')
                
                # Boxplot
                plt.subplot(1, 2, 2)
                sns.boxplot(y=self.df[coluna], color='lightgreen')
                plt.title(f'Boxplot de {coluna}')
                
                plt.tight_layout()
                caminho_arquivo = os.path.join(pasta_graficos, f"{coluna}.png")
                plt.savefig(caminho_arquivo)
                plt.close()
            
            messagebox.showinfo(
                "Concluído",
                f"Análises estatísticas geradas e salvas na pasta '{pasta_graficos}'."
            )
        else:
            messagebox.showwarning("Aviso", "Por favor, importe um arquivo CSV primeiro.")
    
    def executar_modelo(self, modelo_tipo):
        if self.df is not None:
            try:
                # Verificar se 'DIAGNOSTICO' existe
                if 'DIAGNOSTICO' not in self.df.columns:
                    messagebox.showerror("Erro", "A coluna 'DIAGNOSTICO' não foi encontrada no DataFrame.")
                    print("Erro: Coluna 'DIAGNOSTICO' não encontrada.")
                    return
                
                # Preparar dados
                colunas_para_remover = ['ID_PACIENTE', 'DT_COLETA', 'Unnamed: 0']
                X = self.df.drop(colunas_para_remover + ['DIAGNOSTICO'], axis=1, errors='ignore')
                y = self.df['DIAGNOSTICO']
                
                # Verificar se há colunas restantes
                if X.empty:
                    messagebox.showerror("Erro", "Nenhuma coluna disponível para treinamento após a remoção.")
                    print("Erro: Nenhuma coluna disponível para treinamento.")
                    return
                
                # Tratar valores ausentes
                if X.isnull().sum().sum() > 0:
                    X = X.fillna(X.mean())
                    print("Valores ausentes preenchidos com a média das colunas.")
                else:
                    print("Nenhum valor ausente encontrado nos dados.")
                
                # Codificar a variável target se for categórica
                if y.dtype == 'object' or y.dtype.name == 'category':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    print(f"Classes após codificação: {le.classes_}")
                else:
                    y_encoded = y.values
                    print("Variável alvo não categórica.")
                
                # Verificar se há mais de uma classe após a codificação
                if len(set(y_encoded)) < 2:
                    messagebox.showerror("Erro", "A variável alvo possui menos de duas classes após a codificação.")
                    print("Erro: Variável alvo possui menos de duas classes.")
                    return
                
                # Dividir dados em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                )
                print(f"Dados divididos em treino e teste. Tamanho do treino: {X_train.shape[0]}, Tamanho do teste: {X_test.shape[0]}")
                
                # Selecionar e instanciar o modelo corretamente
                modelos = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                    "SVM": SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
                    "KNN": KNeighborsClassifier(n_neighbors=5)
                }
                
                if modelo_tipo not in modelos:
                    messagebox.showerror("Erro", f"Modelo '{modelo_tipo}' não reconhecido.")
                    print(f"Erro: Modelo '{modelo_tipo}' não reconhecido.")
                    return
                
                modelo = modelos[modelo_tipo]
                print(f"Modelo selecionado: {modelo_tipo}")
                print(f"Parâmetros do modelo: {modelo.get_params()}")
                
                # Escalonar os dados se necessário
                if modelo_tipo in ["SVM", "KNN"]:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    print(f"Dados escalonados com StandardScaler para {modelo_tipo}.")
                else:
                    print(f"Escalonamento de dados não necessário para {modelo_tipo}.")
                
                # Treinar o modelo
                modelo.fit(X_train, y_train)
                print(f"Modelo {modelo_tipo} treinado com sucesso.")
                
                # Previsões
                y_pred = modelo.predict(X_test)
                print(f"Modelo {modelo_tipo} fez previsões no conjunto de teste.")
                
                # Avaliar o modelo
                acc = accuracy_score(y_test, y_pred)
                cls_report = classification_report(y_test, y_pred, zero_division=0)
                conf_matrix = confusion_matrix(y_test, y_pred)
                print(f"Acurácia do {modelo_tipo}: {acc*100:.2f}%")
                
                # Validação Cruzada
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(modelo, X, y_encoded, cv=cv, scoring='accuracy')
                print(f"Validação Cruzada - Acurácia média: {cv_scores.mean()*100:.2f}%, Desvio Padrão: {cv_scores.std()*100:.2f}%")
                
                # Calcular ROC Curve e AUC, se binário
                if modelo_tipo in ["Random Forest", "SVM", "KNN"] and hasattr(modelo, "predict_proba") and len(set(y_encoded)) == 2:
                    y_proba = modelo.predict_proba(X_test)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    print(f"AUC do {modelo_tipo}: {roc_auc:.2f}")
                else:
                    fpr, tpr, thresholds, roc_auc = None, None, None, None
                    if len(set(y_encoded)) != 2:
                        print(f"Curva ROC não calculada para {modelo_tipo} devido a múltiplas classes.")
                    elif not hasattr(modelo, "predict_proba"):
                        print(f"Curva ROC não calculada para {modelo_tipo} porque o modelo não possui predict_proba.")
                
                # Exibir resultados em uma nova janela
                self.exibir_resultados(
                    acc, cls_report, conf_matrix, le if (y.dtype == 'object' or y.dtype.name == 'category') else None,
                    modelo, X.columns, modelo_tipo, fpr, tpr, roc_auc, cv_scores
                )
            
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao executar o modelo {modelo_tipo}.\nErro: {e}")
                print(f"Erro ao executar o modelo {modelo_tipo}: {e}")
        else:
            messagebox.showwarning("Aviso", "Por favor, importe um arquivo CSV primeiro.")
            print("Aviso: Arquivo CSV não foi importado.")
    
    def exibir_resultados(self, acc, cls_report, conf_matrix, label_encoder, modelo, feature_names, modelo_tipo, fpr, tpr, roc_auc, cv_scores):
        # Criar uma nova janela para exibir os resultados
        resultados_win = tk.Toplevel(self.root)
        resultados_win.title(f"Resultados do {modelo_tipo}")
        resultados_win.geometry("800x1400")
        
        # Frame para os textos
        frame_texto = tk.Frame(resultados_win)
        frame_texto.pack(pady=10)
        
        # Exibir o nome do modelo
        lbl_modelo = tk.Label(
            frame_texto,
            text=f"Relatório de Classificação - {modelo_tipo}",
            font=("Arial", 14, "bold")
        )
        lbl_modelo.pack(pady=5)
        
        # Exibir a acurácia
        lbl_acuracia = tk.Label(
            frame_texto,
            text=f"Acurácia: {acc*100:.2f}%",
            font=("Arial", 12)
        )
        lbl_acuracia.pack(pady=5)
        
        # Exibir a Validação Cruzada
        lbl_cv = tk.Label(
            frame_texto,
            text=f"Validação Cruzada (5 folds) - Acurácia média: {cv_scores.mean()*100:.2f}%, Desvio Padrão: {cv_scores.std()*100:.2f}%",
            font=("Arial", 12, "bold")
        )
        lbl_cv.pack(pady=5)
        
        # Exibir o relatório de classificação
        lbl_relatorio = tk.Label(
            frame_texto,
            text="Relatório de Classificação:",
            font=("Arial", 12, "bold")
        )
        lbl_relatorio.pack(pady=5)
        
        txt_relatorio = tk.Text(frame_texto, height=10, width=90)
        txt_relatorio.pack()
        txt_relatorio.insert(tk.END, cls_report)
        txt_relatorio.config(state=tk.DISABLED)
        
        # Exibir a matriz de confusão
        lbl_matriz = tk.Label(
            frame_texto,
            text="Matriz de Confusão:",
            font=("Arial", 12, "bold")
        )
        lbl_matriz.pack(pady=5)
        
        txt_matriz = tk.Text(frame_texto, height=10, width=90)
        txt_matriz.pack()
        txt_matriz.insert(tk.END, conf_matrix)
        txt_matriz.config(state=tk.DISABLED)
        
        # Plotar a matriz de confusão
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Previsões')
        ax.set_ylabel('Verdadeiros')
        ax.set_title('Matriz de Confusão')
        
        # Salvar a figura em um arquivo temporário
        plt.tight_layout()
        pasta_graficos = "graficos_analise"
        if not os.path.exists(pasta_graficos):
            os.makedirs(pasta_graficos)
        temp_plot_path = os.path.join(pasta_graficos, f"matriz_confusao_{modelo_tipo}.png")
        plt.savefig(temp_plot_path)
        plt.close()
        
        # Exibir a imagem da matriz de confusão na janela
        try:
            img = Image.open(temp_plot_path)
            # Verificar se 'Resampling' está disponível (Pillow >=10.0.0)
            if hasattr(Image, 'Resampling'):
                img = img.resize((500, 400), Image.Resampling.LANCZOS)
            else:
                img = img.resize((500, 400), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            lbl_imagem = tk.Label(resultados_win, image=photo)
            lbl_imagem.image = photo  # Manter uma referência
            lbl_imagem.pack(pady=10)
        except AttributeError as e:
            messagebox.showerror("Erro", f"Falha ao carregar a imagem da matriz de confusão.\nErro: {e}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar a imagem da matriz de confusão.\nErro: {e}")
        
        # Exibir as classes (se label encoder foi usado)
        if label_encoder:
            classes = label_encoder.classes_
            lbl_classes = tk.Label(
                frame_texto,
                text=f"Classes: {', '.join(classes)}",
                font=("Arial", 12)
            )
            lbl_classes.pack(pady=5)
        
        # Plotar a importância das features para Random Forest apenas
        if isinstance(modelo, RandomForestClassifier):
            fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
            importances = modelo.feature_importances_
            sorted_indices = importances.argsort()[::-1]
            sns.barplot(x=importances[sorted_indices], y=[feature_names[i] for i in sorted_indices], ax=ax_feat, palette='viridis', orient='h')
            ax_feat.set_title('Importância das Features')
            ax_feat.set_xlabel('Importância')
            ax_feat.set_ylabel('Features')
            
            # Salvar e exibir a figura
            plt.tight_layout()
            temp_feat_path = os.path.join(pasta_graficos, f"importancia_features_{modelo_tipo}.png")
            plt.savefig(temp_feat_path)
            plt.close()
            
            try:
                img_feat = Image.open(temp_feat_path)
                if hasattr(Image, 'Resampling'):
                    img_feat = img_feat.resize((500, 400), Image.Resampling.LANCZOS)
                else:
                    img_feat = img_feat.resize((500, 400), Image.ANTIALIAS)
                photo_feat = ImageTk.PhotoImage(img_feat)
                lbl_imagem_feat = tk.Label(resultados_win, image=photo_feat)
                lbl_imagem_feat.image = photo_feat  # Manter uma referência
                lbl_imagem_feat.pack(pady=10)
            except AttributeError as e:
                messagebox.showerror("Erro", f"Falha ao carregar a imagem da importância das features.\nErro: {e}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar a imagem da importância das features.\nErro: {e}")
        
        # Plotar a Curva ROC
        if modelo_tipo in ["Random Forest", "SVM", "KNN"] and fpr is not None and tpr is not None and roc_auc is not None:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            sns.lineplot(x=fpr, y=tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'Curva ROC - {modelo_tipo}')
            ax_roc.legend(loc="lower right")
            
            # Salvar a figura da Curva ROC
            plt.tight_layout()
            temp_roc_path = os.path.join(pasta_graficos, f"curva_roc_{modelo_tipo}.png")
            plt.savefig(temp_roc_path)
            plt.close()
            
            # Exibir a imagem da Curva ROC na janela
            try:
                img_roc = Image.open(temp_roc_path)
                if hasattr(Image, 'Resampling'):
                    img_roc = img_roc.resize((500, 400), Image.Resampling.LANCZOS)
                else:
                    img_roc = img_roc.resize((500, 400), Image.ANTIALIAS)
                photo_roc = ImageTk.PhotoImage(img_roc)
                lbl_roc = tk.Label(resultados_win, image=photo_roc)
                lbl_roc.image = photo_roc  # Manter uma referência
                lbl_roc.pack(pady=10)
            except AttributeError as e:
                messagebox.showerror("Erro", f"Falha ao carregar a imagem da Curva ROC.\nErro: {e}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar a imagem da Curva ROC.\nErro: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicativoAnaliseDados(root)
    root.mainloop()
