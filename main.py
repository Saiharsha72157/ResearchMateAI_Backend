from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from groq import Groq
import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
from services.datasets import DatasetService
from services.auth import get_current_user
from routes import paraphrase_router

load_dotenv()

dataset_service = DatasetService()

app = FastAPI(title="ResearchMateAI Backend", version="1.0.0")

# CORS middleware to allow physical devices and emulators to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modular local routers
app.include_router(paraphrase_router)

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "running"
    }

# Groq Client setup (retains Title Generation capabilities)
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    logger.info("Groq API key loaded successfully.")
else:
    logger.warning("Groq API key is missing from environment variables.")

try:
    if groq_api_key:
        client = Groq(api_key=groq_api_key)
        logger.info("Groq client initialized successfully.")
    else:
        client = None
except Exception as e:
    logger.error(f"[Backend] Groq API Initialization Warning: {e}", exc_info=True)
    client = None

logger.info("Application startup completed.")

# Request Models
class ResearchRequest(BaseModel):
    department: str
    domain: str

class ManualDataRequest(BaseModel):
    column_names: List[str]
    rows: List[List[Any]]

class ChartRegenRequest(BaseModel):
    groups: List[str]
    parameters: List[str]
    comparison_stats: Dict[str, Dict[str, Dict[str, Any]]]
    group_col: str
    title: str
    xlabel: str
    ylabel: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "ResearchMateAI Backend is running"
    }

# AI Title Generation Route (retains existing functionality)
@app.post("/generate-titles")
def generate_titles(data: ResearchRequest, current_user: Dict[str, Any] = Depends(get_current_user)):

    # Fallback generator in case of exceptions or empty results
    def get_fallback_projects(dept: str, dom: str):
        fallbacks = {
            "Artificial Intelligence": [
                {
                    "title": "Multimodal Explainable AI for Real-World Decision Support Systems",
                    "difficulty": "Hard",
                    "algorithms": ["Transformers", "SHAP", "LIME"],
                    "summary": "This project builds a multi-modal AI decision-support system that processes textual descriptions, raw visual scans, and tabular features, yielding fully explainable, audit-ready clinical decisions.",
                    "dataset": "MIMIC-IV Multi-modal Intensive Care Dataset (clinical notes, tables, X-rays).",
                    "best_algorithms_explanation": "Multimodal Transformers are best because self-attention integrates diverse modalities natively. SHAP and LIME are best because they generate mathematical feature attribution scores to guarantee transparency."
                },
                {
                    "title": "AI-Powered Traffic Prediction System Using Deep Learning",
                    "difficulty": "Medium",
                    "algorithms": ["LSTM", "CNN", "Random Forest"],
                    "summary": "A city-scale predictive grid that leverages neural structures to anticipate traffic congestions and compute dynamic navigation maps.",
                    "dataset": "PeMS-SF traffic sensor flow database from Caltrans (California Performance Measurement System).",
                    "best_algorithms_explanation": "LSTMs are best because they capture long-range chronological traffic correlations, and CNNs are best if spatial network relationships are represented as 2D spatial matrices."
                },
                {
                    "title": "Autonomous Robotics Pathfinding using Deep Q-Networks",
                    "difficulty": "Hard",
                    "algorithms": ["DQN", "A*", "DDPG"],
                    "summary": "Implement a self-learning navigation loop where a robot agent optimizes continuous routing through obstacle-heavy environments.",
                    "dataset": "CARLA Autonomous Driving Simulator lidar and camera telemetry feeds.",
                    "best_algorithms_explanation": "Deep Q-Networks (DQN) are best because they project raw pixel inputs directly into optimal movement policies without requiring manual spatial modeling."
                },
                {
                    "title": "AI-Driven Patient Monitoring and Diagnostic Assistant",
                    "difficulty": "Medium",
                    "algorithms": ["ResNet", "Random Forest", "XGBoost"],
                    "summary": "Continuously evaluate clinical patient parameters to proactively detect and report cardiovascular anomalies.",
                    "dataset": "PhysioNet MIMIC clinical database and historical ECG telemetry records.",
                    "best_algorithms_explanation": "ResNet is best for robust clinical image classification, and XGBoost operates optimally on structured tabular clinical vitals to identify anomalies."
                },
                {
                    "title": "Smart Home Automation with IoT and Machine Learning",
                    "difficulty": "Easy",
                    "algorithms": ["Decision Trees", "K-Means", "Linear Regression"],
                    "summary": "An intelligent controller that learns household behavior patterns to maximize energy conservation in cooling systems.",
                    "dataset": "Smart Home Environmental Sensors Dataset (temperatures, motion sensors, occupancy indicators).",
                    "best_algorithms_explanation": "Decision Trees are ideal because they construct easy-to-read logical control rules, and K-Means clusters hourly activities to formulate custom appliance schedules."
                }
            ],
            "Machine Learning": [
                {
                    "title": "Evaluating the Efficacy of Reinforcement Learning for Autonomous Navigation",
                    "difficulty": "Hard",
                    "algorithms": ["PPO", "DDPG", "SAC"],
                    "summary": "Evaluate state-of-the-art policy optimization schemes under complex navigation constraints.",
                    "dataset": "ROS Gazebo Physical Robot Simulator telemetry traces.",
                    "best_algorithms_explanation": "Proximal Policy Optimization (PPO) is best because it features clipped objective targets, preventing destructive model divergence during robot control training."
                },
                {
                    "title": "Predictive Maintenance System for Industrial Equipment",
                    "difficulty": "Medium",
                    "algorithms": ["SVM", "Gradient Boosting", "Neural Networks"],
                    "summary": "Analyze high-frequency vibrational readings to anticipate thermal anomalies and structural degradation in turbines.",
                    "dataset": "NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS).",
                    "best_algorithms_explanation": "Gradient Boosting (XGBoost/LightGBM) is best because it handles highly skewed industrial sensor data and creates precise predictive anomaly targets."
                },
                {
                    "title": "Real-Time Sentiment Analysis for Social Media Platforms",
                    "difficulty": "Hard",
                    "algorithms": ["BERT", "Transformer", "RNN"],
                    "summary": "Capture complex emotional transitions and semantic nuances in stream-based social conversations.",
                    "dataset": "Sentiment140 Twitter Sentiment dataset (1.6 million annotated tweets).",
                    "best_algorithms_explanation": "BERT is best because its bidirectional attention models complex sentence semantics, capturing implicit context, slang, and user sarcasm perfectly."
                },
                {
                    "title": "Customer Churn Prediction using Ensemble Classifiers",
                    "difficulty": "Easy",
                    "algorithms": ["Random Forest", "Logistic Regression", "XGBoost"],
                    "summary": "Predict user churn probabilities based on transaction frequencies and historical user profiles.",
                    "dataset": "Kaggle Telco Customer Churn database.",
                    "best_algorithms_explanation": "Random Forest is excellent because it averages multiple weak decision trees, avoiding overfitting and naturally handling categorical properties."
                }
            ]
        }
        
        default_projects = [
            {
                "title": f"Next-Gen {dom} framework for {dept} Applications",
                "difficulty": "Medium",
                "algorithms": ["Random Forest", "XGBoost", "K-Means"],
                "summary": f"Develop an advanced data-driven framework leveraging state-of-the-art algorithms within the {dom} domain.",
                "dataset": f"Public research datasets related to {dom} and engineering systems.",
                "best_algorithms_explanation": "Random Forest and XGBoost are best because they offer high accuracy and robust feature importances for structured metrics."
            },
            {
                "title": f"Automated Anomaly Detection in {dom} Environments",
                "difficulty": "Hard",
                "algorithms": ["Isolation Forest", "Autoencoders", "SVM"],
                "summary": "A real-time monitoring tool designed to capture micro-anomalies and out-of-distribution patterns.",
                "dataset": "Synthetic and physical network telemetry logs.",
                "best_algorithms_explanation": "Autoencoders are best because they learn standard patterns unsupervised and isolate outliers based on high reconstruction error."
            },
            {
                "title": f"Predictive Modeling and Optimization for {dept} Systems",
                "difficulty": "Medium",
                "algorithms": ["Linear Regression", "Gradient Boosting", "KNN"],
                "summary": f"Optimize the performance and throughput of system architectures using statistical modeling.",
                "dataset": f"Historical performance indicators and sensor readings of {dept} hardware.",
                "best_algorithms_explanation": "Gradient Boosting is best for capturing complex, non-linear relationships between hardware parameters."
            },
            {
                "title": f"Edge-Computing Optimization for {dom}-Enabled IoT Nodes",
                "difficulty": "Hard",
                "algorithms": ["Quantized CNN", "SVM", "Decision Trees"],
                "summary": "Deploy deep learning models on low-power edge microcontrollers through post-training weight quantization.",
                "dataset": "UCI Smart Buildings and sensor monitoring data.",
                "best_algorithms_explanation": "Quantized CNNs are best because they compress the weight parameters, fitting model parameters on edge hardware storage."
            },
            {
                "title": f"Statistical Analysis and Clustering of {dom} Parameters",
                "difficulty": "Easy",
                "algorithms": ["K-Means", "PCA", "Hierarchical Clustering"],
                "summary": f"Explore and categorize dynamic metrics across multiple {dept} domains to discover hidden archetypes.",
                "dataset": "Public tabular metrics datasets.",
                "best_algorithms_explanation": "K-Means combined with Principal Component Analysis (PCA) is best because it reduces feature dimensionality for simple, intuitive clustering."
            }
        ]
        return fallbacks.get(dom, default_projects)

    if not client:
        print("[Backend] Groq not configured, returning mock projects.")
        return {"projects": get_fallback_projects(data.department, data.domain)}
    
    try:
        prompt = f"""
        Generate exactly 5 innovative, highly professional, and realistic academic research project titles and detailed plans for:

        Department: {data.department}
        Domain: {data.domain}

        For each project, generate:
        1. "title": A highly formal, academic, and IEEE-standard research paper title (do not use quotes). The title must be framed exactly like a publication in an IEEE Transactions or IEEE Conference journal. Avoid casual or generic phrases. Use formal academic structures such as "A [Mechanism/Framework] for [Goal/Application] Using [Algorithms]", "Performance Analysis of [Approach] in [Problem Domain]", or "Deep Learning-Based [System] for [Task]: A Comparative Evaluation".
        2. "difficulty": A difficulty level ('Easy', 'Medium', or 'Hard').
        3. "algorithms": A list of 2-3 suggested machine learning or data science algorithms (e.g., ['LSTM', 'CNN', 'Random Forest']).
        4. "summary": A clear, academically robust, and compelling summary of what the project accomplishes.
        5. "dataset": What dataset to use for this project (specify professional public datasets like Kaggle, UCI, PhysioNet, ImageNet, PeMS-SF, etc.).
        6. "best_algorithms_explanation": Tell me what algorithms are best and why they are best for this specific project.

        Return ONLY a valid JSON object matching this schema (do not include any additional text or markdown formatting):
        {{
            "projects": [
                {{
                    "title": "compelling formal IEEE project title",
                    "difficulty": "Easy" | "Medium" | "Hard",
                    "algorithms": ["algorithm1", "algorithm2"],
                    "summary": "detailed summary here...",
                    "dataset": "recommended dataset here...",
                    "best_algorithms_explanation": "why these algorithms are ideal..."
                }},
                ...
            ]
        }}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        import json
        result = response.choices[0].message.content
        data_json = json.loads(result)
        projects = data_json.get("projects", [])
        
        if not projects and isinstance(data_json, list):
            projects = data_json

        clean_projects = []
        for proj in projects[:5]:
            title = proj.get("title", "").strip().replace('"', '')
            difficulty = proj.get("difficulty", "Medium")
            if difficulty not in ["Easy", "Medium", "Hard"]:
                difficulty = "Medium"
            algorithms = proj.get("algorithms", [])
            if not isinstance(algorithms, list):
                algorithms = [str(algorithms)]
            summary = proj.get("summary", "")
            dataset = proj.get("dataset", "")
            best_explanation = proj.get("best_algorithms_explanation", "")
            
            if title:
                clean_projects.append({
                    "title": title,
                    "difficulty": difficulty,
                    "algorithms": [alg.strip() for alg in algorithms if alg],
                    "summary": summary.strip(),
                    "dataset": dataset.strip(),
                    "best_algorithms_explanation": best_explanation.strip()
                })

        if not clean_projects:
            raise ValueError("No valid projects generated")

        return {
            "projects": clean_projects
        }
    except Exception as e:
        print(f"[Backend] Error generating titles: {e}, falling back to mock projects.")
        return {"projects": get_fallback_projects(data.department, data.domain)}

# Helper to clean numbers for JSON serialization
def clean_float(val: Any) -> Optional[float]:
    if pd.isnull(val) or (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
        return None
    return float(val)

def clean_int(val: Any) -> Optional[int]:
    if pd.isnull(val) or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return None
    return int(val)

# Core analysis logic to calculate comparative statistics
def analyze_dataframe(df: pd.DataFrame, file_name: str, source: str = "csv_upload") -> Dict[str, Any]:
    try:
        if df.empty:
            raise ValueError("The dataset is empty. Please upload a file containing data rows.")

        rows, columns = df.shape
        column_names = [str(col) for col in df.columns.tolist()]
        
        # Identify numeric and categorical columns
        numeric_columns = []
        categorical_columns = []
        for col in df.columns:
            col_str = str(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col_str)
            else:
                categorical_columns.append(col_str)

        # 1. Identify primary grouping column (categorical with 2-5 unique values)
        group_col = None
        for col in categorical_columns:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 5:
                group_col = col
                break
        
        if not group_col and categorical_columns:
            # Fallback to first categorical column if it has at least 2 unique values
            for col in categorical_columns:
                if df[col].nunique() >= 2:
                    group_col = col
                    break
        
        # Determine groups
        if group_col:
            groups = [str(g) for g in df[group_col].dropna().unique().tolist()]
        else:
            group_col = "Group"
            groups = ["All Data"]

        # Helper to clean numbers with high precision (4 decimal places)
        def round_precision(val: Any) -> Optional[float]:
            if pd.isnull(val) or (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
                return None
            return float(round(val, 4))

        # Calculate statistics: mean, n, std, sem
        comparison_stats = {}
        
        # For grouped data
        if groups != ["All Data"] and group_col:
            for g in groups:
                comparison_stats[g] = {}
                g_df = df[df[group_col] == g]
                for col in numeric_columns:
                    col_data = g_df[col].dropna()
                    n = int(col_data.count())
                    mean = round_precision(col_data.mean())
                    std = round_precision(col_data.std())
                    sem = round_precision(col_data.sem()) if n > 1 else 0.0
                    
                    comparison_stats[g][col] = {
                        "n": n,
                        "mean": mean,
                        "std": std,
                        "sem": sem
                    }
        else:
            # For single group
            comparison_stats["All Data"] = {}
            for col in numeric_columns:
                col_data = df[col].dropna()
                n = int(col_data.count())
                mean = round_precision(col_data.mean())
                std = round_precision(col_data.std())
                sem = round_precision(col_data.sem()) if n > 1 else 0.0
                
                comparison_stats["All Data"][col] = {
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "sem": sem
                }

        # 2. Generate comparison grouped bar chart
        graph_base64 = None
        if len(numeric_columns) > 0:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Parameters (X-axis labels)
                parameters = numeric_columns[:10]  # Show up to 10 parameters
                x = np.arange(len(parameters))
                width = 0.8 / len(groups)  # Dynamic bar width based on number of groups
                
                for idx, g in enumerate(groups):
                    means = []
                    sems = []
                    for param in parameters:
                        p_stats = comparison_stats[g].get(param, {"mean": 0.0, "sem": 0.0})
                        means.append(p_stats.get("mean") or 0.0)
                        sems.append(p_stats.get("sem") or 0.0)
                    
                    # Plot bars with error bars representing SEM
                    ax.bar(
                        x + (idx - len(groups)/2 + 0.5) * width, 
                        means, 
                        width, 
                        yerr=sems, 
                        label=g, 
                        capsize=4,
                        alpha=0.85
                    )
                
                ax.set_title(f"Comparison across Parameters (by {group_col})", fontsize=14, fontweight='bold', pad=15)
                ax.set_xticks(x)
                ax.set_xticklabels(parameters, rotation=15, ha='right', fontsize=10)
                ax.set_ylabel("Value", fontsize=11)
                ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
                
                # Remove top and right spines
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                graph_base64 = f"data:image/png;base64,{img_str}"
            except Exception as chart_err:
                print(f"[Backend] Error generating comparison bar chart: {chart_err}")

        # Construct final response
        return {
            "success": True,
            "file_name": file_name,
            "source": source,
            "rows": rows,
            "columns": columns,
            "group_col": group_col,
            "groups": groups,
            "parameters": numeric_columns,
            "comparison_stats": comparison_stats,
            "comparison_graph": graph_base64
        }
    except Exception as e:
        print(f"[Backend] Processing error: {e}")
        raise ValueError(str(e))

# CSV File analysis endpoint
@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...), current_user: Dict[str, Any] = Depends(get_current_user)):

    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a valid CSV file (.csv)."
        )
    
    try:
        # Read uploaded file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400, 
                detail="The uploaded file is empty. Please upload a file with data."
            )
            
        # Parse CSV using Pandas
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as parse_error:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to parse CSV file: {str(parse_error)}"
            )

        # Run analysis pipeline
        result = analyze_dataframe(df, file_name=file.filename, source="csv_upload")
        return result

    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"[Backend] Server error in /analyze-csv: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while analyzing the CSV: {str(e)}"
        )

# Manual data analysis endpoint (Bonus!)
@app.post("/analyze-manual-data")
def analyze_manual_data(data: ManualDataRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    if not data.column_names or not data.rows:
        raise HTTPException(
            status_code=400,
            detail="Columns and rows must not be empty."
        )

    try:
        # Create DataFrame from 2D rows and columns
        try:
            df = pd.DataFrame(data.rows, columns=data.column_names)
        except Exception as df_error:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to structure data: {str(df_error)}"
            )

        result = analyze_dataframe(df, file_name="Manual Entry Data", source="manual_entry")
        return result

    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"[Backend] Server error in /analyze-manual-data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing the data: {str(e)}"
        )

# Regenerate comparative grouped bar chart with custom labels
@app.post("/regenerate-chart")
def regenerate_chart(data: ChartRegenRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        groups = data.groups
        parameters = data.parameters[:10]  # Show up to 10 parameters
        x = np.arange(len(parameters))
        width = 0.8 / len(groups)
        
        for idx, g in enumerate(groups):
            means = []
            sems = []
            for param in parameters:
                p_stats = data.comparison_stats.get(g, {}).get(param, {"mean": 0.0, "sem": 0.0})
                means.append(p_stats.get("mean") or 0.0)
                sems.append(p_stats.get("sem") or 0.0)
            
            # Plot bars with error bars representing SEM
            ax.bar(
                x + (idx - len(groups)/2 + 0.5) * width, 
                means, 
                width, 
                yerr=sems, 
                label=g, 
                capsize=4,
                alpha=0.85
            )
        
        ax.set_title(data.title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(parameters, rotation=15, ha='right', fontsize=10)
        ax.set_xlabel(data.xlabel, fontsize=11)
        ax.set_ylabel(data.ylabel, fontsize=11)
        
        # Only show legend if there are actually multiple groups
        if groups != ["All Data"]:
            ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
        
        # Remove top and right spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "success": True,
            "comparison_graph": f"data:image/png;base64,{img_str}"
        }
    except Exception as e:
        print(f"[Backend] Error regenerating chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dataset Search Route
@app.get("/search-datasets")
def search_datasets(query: Optional[str] = "ai", provider: Optional[str] = "kaggle", page: Optional[int] = 1, limit: Optional[int] = 20, current_user: Dict[str, Any] = Depends(get_current_user)):

    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")
    try:
        results = dataset_service.search_all(query=query, provider=provider, page=page, limit=limit)
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"[Backend] Error searching datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching datasets: {str(e)}")


