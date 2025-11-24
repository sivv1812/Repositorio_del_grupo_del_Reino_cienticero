import gradio as gr
import requests
import os

# URL interna para conectarse al backend
API_URL = os.getenv("API_URL", "http://backend:8000/predict")

def predecir_compra(entregas, visitas, tipo_cliente, tamano, categoria, marca):
    payload = {
        "num_deliver_per_week": float(entregas),
        "num_visit_per_week": float(visitas),
        "customer_type": tipo_cliente,
        "size": tamano,
        "category": categoria,
        "brand": marca
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            return f"{result['message']}\n(Probabilidad: {result['probability']:.2%})"
        else:
            return f"Error del servidor: {response.text}"
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# Diseño de la interfaz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🥤 SodAI Drinks - Predicción de Ventas")
    gr.Markdown("Ingrese los datos del cliente para predecir si realizará una compra la próxima semana.")
    
    with gr.Row():
        with gr.Column():
            num_deliver = gr.Slider(0, 50, value=1, label="Entregas por semana", step=1)
            num_visit = gr.Slider(0, 50, value=1, label="Visitas por semana", step=1)
            cust_type = gr.Dropdown(["type_1", "type_2", "type_3", "type_4"], label="Tipo de Cliente", value="type_1")
        
        with gr.Column():
            size = gr.Dropdown(["small", "medium", "large", "extra_large"], label="Tamaño Producto", value="medium")
            category = gr.Dropdown(["water", "soda", "juice", "energy"], label="Categoría", value="soda")
            brand = gr.Dropdown(["brand_A", "brand_B", "brand_C", "brand_D"], label="Marca", value="brand_A")
            
    btn = gr.Button("🔮 Predecir Compra", variant="primary")
    resultado = gr.Textbox(label="Resultado del Modelo", lines=2)
    
    btn.click(predecir_compra, 
              inputs=[num_deliver, num_visit, cust_type, size, category, brand], 
              outputs=resultado)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)