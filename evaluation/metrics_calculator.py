import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, df: pd.DataFrame):
        """
        Expects a DataFrame with 'true_labels' and 'predicted_labels'.
        Both columns must contain lists of strings (Technique IDs).
        """
        self.df = df
        self.mlb = MultiLabelBinarizer()
        
    def evaluate(self) -> dict:
        if 'true_labels' not in self.df.columns or 'predicted_labels' not in self.df.columns:
            raise ValueError("DataFrame must contain 'true_labels' and 'predicted_labels'")
            
        # Fit binarizer on all possible labels (true + predicted)
        all_labels = self.df['true_labels'].tolist() + self.df['predicted_labels'].tolist()
        self.mlb.fit(all_labels)
        
        y_true = self.mlb.transform(self.df['true_labels'])
        y_pred = self.mlb.transform(self.df['predicted_labels'])
        
        # Calculate standard Micro and Macro (Precision, Recall, F1)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        # Calculate F0.5 (Prioritize Precision over Recall, crucial for CTI)
        f05_micro = fbeta_score(y_true, y_pred, beta=0.5, average='micro', zero_division=0)
        f05_macro = fbeta_score(y_true, y_pred, beta=0.5, average='macro', zero_division=0)
        
        # Per-class metrics
        p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        class_report = []
        for i, class_name in enumerate(self.mlb.classes_):
            # Only report classes that had some activity (either in True or Pred)
            if support[i] > 0 or sum(y_pred[:, i]) > 0:
                class_report.append({
                    "technique": class_name,
                    "precision": round(float(p_class[i]), 4),
                    "recall": round(float(r_class[i]), 4),
                    "f1": round(float(f1_class[i]), 4),
                    "support": int(support[i])  # CRITICAL FIX FOR JSON SERIALIZATION
                })
        
        results = {
            "micro": {
                "precision": round(p_micro, 4), "recall": round(r_micro, 4),
                "f1": round(f1_micro, 4), "f0.5": round(f05_micro, 4)
            },
            "macro": {
                "precision": round(p_macro, 4), "recall": round(r_macro, 4),
                "f1": round(f1_macro, 4), "f0.5": round(f05_macro, 4)
            },
            "class_report": sorted(class_report, key=lambda x: x['support'], reverse=True)
        }
        
        return results
