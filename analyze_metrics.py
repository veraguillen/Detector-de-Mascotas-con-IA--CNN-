import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def load_latest_training_results(results_dir: str = 'results') -> Dict[str, Any]:
    """Carga los resultados del entrenamiento más reciente."""
    # Obtener todos los directorios de resultados
    training_dirs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('training_')]
    
    if not training_dirs:
        raise FileNotFoundError("No se encontraron directorios de entrenamiento en la carpeta 'results'")
    
    # Ordenar por fecha (el más reciente primero)
    latest_dir = sorted(training_dirs, reverse=True)[0]
    results_path = os.path.join(results_dir, latest_dir)
    
    print(f"Cargando resultados de: {results_path}")
    
    # Cargar datos
    results = {
        'directory': results_path,
        'config': json.load(open(os.path.join(results_path, 'training_config.json'), encoding='utf-8')),
        'history': pd.read_csv(os.path.join(results_path, 'training_history.csv')),
        'classification_report': json.load(open(os.path.join(results_path, 'classification_report.json'), encoding='utf-8'))
    }
    
    # Cargar matriz de confusión si existe
    confusion_matrix_path = os.path.join(results_path, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        results['confusion_matrix_path'] = confusion_matrix_path
    
    return results

def plot_metrics_comparison(history: pd.DataFrame, output_dir: str = None):
    """Genera gráficos comparativos de las métricas de entrenamiento y validación."""
    metrics = [col.replace('val_', '') for col in history.columns if col.startswith('val_')]
    
    for metric in metrics:
        train_metric = metric
        val_metric = f'val_{metric}'
        
        if val_metric in history.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(history[train_metric], label=f'Entrenamiento {metric}')
            plt.plot(history[val_metric], label=f'Validación {metric}')
            
            plt.title(f'Comparación de {metric} - Entrenamiento vs Validación')
            plt.xlabel('Época')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
                plt.close()
            else:
                plt.show()

def plot_class_metrics(classification_report: Dict, output_dir: str = None):
    """Genera gráficos de barras para las métricas por clase."""
    metrics = ['precision', 'recall', 'f1-score']
    classes = [k for k in classification_report.keys() 
              if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']]
    
    # Crear un DataFrame con las métricas
    data = []
    for cls in classes:
        for metric in metrics:
            data.append({
                'Clase': cls,
                'Métrica': metric.capitalize(),
                'Valor': classification_report[cls][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Gráfico de barras agrupado
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Clase', y='Valor', hue='Métrica', data=df)
    plt.title('Métricas por Clase')
    plt.ylim(0, 1.1)
    plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'class_metrics.png'))
        plt.close()
    else:
        plt.show()

def generate_html_report(results: Dict[str, Any], output_file: str = 'report.html'):
    """Genera un informe HTML con todas las métricas."""
    # Asegurarse de que el directorio de salida exista
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Obtener rutas relativas para las imágenes
    def get_relative_path(img_name):
        return os.path.join('results', os.path.basename(results['directory']), img_name)
    
    # Plantilla HTML
    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Entrenamiento</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .metric-card {{ 
                background: #f5f5f5; 
                padding: 15px; 
                border-radius: 5px; 
                flex: 1; 
                min-width: 200px;
            }}
            img {{ max-width: 100%; height: auto; }}
            .row {{ display: flex; gap: 20px; margin-bottom: 20px; }}
            .col {{ flex: 1; }}
            pre {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reporte de Entrenamiento</h1>
            <p><strong>Fecha:</strong> {results['config'].get('timestamp', 'N/A')}</p>
            
            <div class="section">
                <h2>Configuración</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Hiperparámetros</h3>
                        <p><strong>Tamaño de imagen:</strong> {results['config'].get('image_size', 'N/A')}</p>
                        <p><strong>Batch size:</strong> {results['config'].get('batch_size', 'N/A')}</p>
                        <p><strong>Learning rate:</strong> {results['config'].get('learning_rate', 'N/A')}</p>
                        <p><strong>Épocas:</strong> {results['config'].get('epochs', 'N/A')}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Clases</h3>
                        {''.join([f'<p><strong>{k}:</strong> {v}</p>' 
                                for k, v in results['config'].get('class_indices', {}).items()])}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Métricas de Entrenamiento</h2>
                <div class="row">
                    <div class="col">
                        <h3>Pérdida (Loss)</h3>
                        <img src="{get_relative_path('comparison_loss.png')}" alt="Comparación de pérdida">
                    </div>
                    <div class="col">
                        <h3>Exactitud (Accuracy)</h3>
                        <img src="{get_relative_path('comparison_accuracy.png')}" alt="Comparación de exactitud">
                    </div>
                </div>
                <div class="row">
                    <div class="col">
                        <h3>Precisión (Precision)</h3>
                        <img src="{get_relative_path('comparison_precision.png')}" alt="Comparación de precisión">
                    </div>
                    <div class="col">
                        <h3>Sensibilidad (Recall)</h3>
                        <img src="{get_relative_path('comparison_recall.png')}" alt="Comparación de recall">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Matriz de Confusión</h2>
                <img src="{get_relative_path('confusion_matrix.png')}" alt="Matriz de Confusión" style="max-width: 600px;">
            </div>
            
            <div class="section">
                <h2>Métricas por Clase</h2>
                <img src="class_metrics.png" alt="Métricas por Clase">
            </div>
            
            <div class="section">
                <h2>Reporte de Clasificación</h2>
                <pre>{json.dumps(results['classification_report'], indent=2)}</pre>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(template)

def main():
    try:
        # Cargar resultados del último entrenamiento
        results = load_latest_training_results()
        
        # Crear directorio de análisis
        analysis_dir = 'analysis_results'
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Generar gráficos
        plot_metrics_comparison(results['history'], analysis_dir)
        plot_class_metrics(results['classification_report'], analysis_dir)
        
        # Generar informe HTML
        generate_html_report(results, os.path.join(analysis_dir, 'report.html'))
        
        # Copiar archivos necesarios para el informe
        import shutil
        for img in ['confusion_matrix.png'] + [f'comparison_{m}.png' for m in ['loss', 'accuracy', 'precision', 'recall']]:
            src = os.path.join(results['directory'], img)
            dst = os.path.join(analysis_dir, img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"\nAnálisis completado exitosamente!")
        print(f"Puedes encontrar los resultados en la carpeta: {os.path.abspath(analysis_dir)}")
        print(f"Abre el archivo 'report.html' en tu navegador para ver el informe completo.")
        
    except Exception as e:
        print(f"\nError al generar el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
