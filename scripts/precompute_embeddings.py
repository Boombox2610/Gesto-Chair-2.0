import os
import sys
import pickle
from ultralytics import YOLOE

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cv_pipeline.config import config
from cv_pipeline.vocabulary import OUTDOOR_LABELS, INDOOR_LABELS

def main():
    print(f"[INFO] Loading {config.yoloe_model} to precompute embeddings...")
    try:
        model = YOLOE(config.yoloe_model)
    except Exception as e:
        print(f"[ERROR] Failed to load model {config.yoloe_model}. Are you online? {e}")
        sys.exit(1)
        
    print("[INFO] Computing embeddings for OUTDOOR mode...")
    model.set_classes(OUTDOOR_LABELS)
    outdoor_embeds = model.model.predictor.txt_feats if hasattr(model.model, 'predictor') and hasattr(model.model.predictor, 'txt_feats') else None
    
    print("[INFO] Computing embeddings for INDOOR mode...")
    model.set_classes(INDOOR_LABELS)
    indoor_embeds = model.model.predictor.txt_feats if hasattr(model.model, 'predictor') and hasattr(model.model.predictor, 'txt_feats') else None
    
    # Store just the class lists for now as ultralytics handles the caching internally
    # We create a dummy file just to signal offline mode that we attempted this
    cache = {
        "outdoor": OUTDOOR_LABELS,
        "indoor": INDOOR_LABELS
    }
    
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "yoloe_vocab_embeddings.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
        
    print(f"[SUCCESS] Wrote vocabulary cache to {out_path}")
    print("[INFO] Note: Ultralytics YOLOE uses the CLIP text encoder. If you plan to run entirely offline, ")
    print("ensure that the CLIP weights are downloaded and cached in your HuggingFace/Torch cache directory.")

if __name__ == "__main__":
    main()
