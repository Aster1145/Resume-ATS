import spacy
from spacy.training.example import Example
import random
import mlflow
import mlflow.spacy

# 1. OUR SYNTHETIC DATASET
# We must show the AI exact sentences and point to the mathematical character 
# positions (start_char, end_char) of the entities we want it to learn.
TRAIN_DATA = [
    ("Akshay M is highly skilled in Python and PyTorch.", 
     {"entities": [(0, 8, "PERSON"), (32, 38, "SKILL"), (43, 50, "SKILL")]}),
     
    ("Hari Vishnu built an application using HTML/CSS and Node.js.", 
     {"entities": [(0, 11, "PERSON"), (39, 47, "SKILL"), (52, 59, "SKILL")]}),
     
    ("Expertise includes 3D modeling in Blender and VRay.", 
     {"entities": [(34, 41, "SKILL"), (46, 50, "SKILL")]}),
     
    ("Prajwal works extensively with Microsoft Azure.", 
     {"entities": [(0, 7, "PERSON"), (31, 46, "SKILL")]}),
     
    ("Familiar with drone navigation and Grasshopper scripting.", 
     {"entities": [(35, 46, "SKILL")]})
]

def train_custom_model():
    print("Initializing blank AI model...")
    # Create a blank English model
    nlp = spacy.blank("en")
    
    # Add the Named Entity Recognizer (NER) to the model's pipeline
    ner = nlp.add_pipe("ner")
    
    # Tell the NER what labels it should be looking for
    ner.add_label("PERSON")
    ner.add_label("SKILL")
    
    # Format the data for spaCy
    examples = []
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annotations))
        
    # 2. THE DEEP LEARNING TRAINING LOOP
    print("Beginning training loop (10 iterations)...")
    optimizer = nlp.begin_training()
    
    with mlflow.start_run() as run:
        print(f"Started MLflow Run: {run.info.run_id}")
        mlflow.log_param("iterations", 10)
        mlflow.log_param("dropout_rate", 0.5)
        
        for i in range(10):
            random.shuffle(examples)
            losses = {}
            # Update the neural network weights based on our examples
            nlp.update(examples, drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {i+1} Loss: {losses}")
            
            mlflow.log_metric("ner_loss", losses.get('ner', 0.0), step=i+1)
            
        # 3. SAVE THE NEW BRAIN
        output_dir = "my_custom_resume_model"
        nlp.to_disk(output_dir)
        print(f"\nTraining complete! Custom AI saved to folder: '{output_dir}'")
        
        # 4. TRACK MODEL WITH MLFLOW
        mlflow.spacy.log_model(spacy_model=nlp, artifact_path="model")
        print("\n✅ spaCy Model tracked and saved to MLflow successfully!")

if __name__ == "__main__":
    train_custom_model()