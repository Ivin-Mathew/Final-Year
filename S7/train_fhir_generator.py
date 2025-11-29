import pandas as pd
import gzip
import json
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
try:
    from fhir.resources.bundle import Bundle
    FHIR_VALIDATION = True
except ImportError:
    FHIR_VALIDATION = False
    print("fhir.resources not available; skipping FHIR validation.")

# Constants
MIMIC_PATH = './mimic-iv-clinical-database-demo-2.2'
FILES = {
    'patients': 'hosp/patients.csv.gz',
    'admissions': 'hosp/admissions.csv.gz',
    'diagnoses': 'hosp/diagnoses_icd.csv.gz',
    'prescriptions': 'hosp/prescriptions.csv.gz'
}
MODEL_NAME = 'distilgpt2'
OUTPUT_DIR = './fhir_model'

def load_data():
    """Load MIMIC-IV CSV files into DataFrames."""
    data = {}
    for key, file in FILES.items():
        path = os.path.join(MIMIC_PATH, file)
        try:
            with gzip.open(path, 'rt') as f:
                data[key] = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
    return data

def map_to_fhir_bundle(patient_row, admissions, diagnoses, prescriptions):
    """Map patient data to a FHIR Bundle JSON."""
    subject_id = patient_row['subject_id']
    
    # Patient resource (anonymized)
    patient = {
        "resourceType": "Patient",
        "id": f"patient-{subject_id}",
        "gender": patient_row['gender'],
        "birthDate": patient_row['anchor_age'].astype(str) + "-01-01" if pd.notna(patient_row['anchor_age']) else None  # Generalized age
    }
    
    # Encounters from admissions
    encounters = []
    pat_admissions = admissions[admissions['subject_id'] == subject_id]
    for _, adm in pat_admissions.iterrows():
        encounter = {
            "resourceType": "Encounter",
            "id": f"encounter-{adm['hadm_id']}",
            "subject": {"reference": f"Patient/{subject_id}"},
            "period": {
                "start": adm['admittime'],
                "end": adm['dischtime']
            },
            "class": {"code": "IMP"}  # Inpatient
        }
        encounters.append(encounter)
    
    # Conditions from diagnoses
    conditions = []
    pat_diagnoses = diagnoses[diagnoses['subject_id'] == subject_id]
    for _, diag in pat_diagnoses.iterrows():
        condition = {
            "resourceType": "Condition",
            "id": f"condition-{diag['hadm_id']}-{diag['icd_code']}",
            "subject": {"reference": f"Patient/{subject_id}"},
            "code": {"coding": [{"code": diag['icd_code'], "display": diag['icd_title']}]}
        }
        conditions.append(condition)
    
    # MedicationRequests from prescriptions
    meds = []
    pat_prescriptions = prescriptions[prescriptions['subject_id'] == subject_id]
    for _, presc in pat_prescriptions.iterrows():
        med = {
            "resourceType": "MedicationRequest",
            "id": f"medreq-{presc['hadm_id']}-{presc['drug']}",
            "subject": {"reference": f"Patient/{subject_id}"},
            "medicationCodeableConcept": {"text": presc['drug']}
        }
        meds.append(med)
    
    # Bundle
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": patient}
        ] + [{"resource": enc} for enc in encounters] + [{"resource": cond} for cond in conditions] + [{"resource": med} for med in meds]
    }
    
    if FHIR_VALIDATION:
        try:
            Bundle(**bundle)  # Validate
        except Exception as e:
            print(f"Invalid FHIR Bundle for patient {subject_id}: {e}")
            return None
    return bundle

def preprocess_data(data):
    """Preprocess data and create serialized FHIR Bundles."""
    patients = data['patients']
    admissions = data['admissions']
    diagnoses = data['diagnoses']
    prescriptions = data['prescriptions']
    
    bundles = []
    for _, patient in patients.iterrows():
        bundle = map_to_fhir_bundle(patient, admissions, diagnoses, prescriptions)
        if bundle:
            bundles.append(json.dumps(bundle))
    return bundles

def main():
    # Load data
    data = load_data()
    
    # Preprocess
    bundles = preprocess_data(data)
    if not bundles:
        raise ValueError("No valid bundles generated.")
    
    # Create Dataset
    dataset = Dataset.from_dict({"text": bundles})
    train_val, test = train_test_split(dataset, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)
    
    # Tokenize
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    train = train.map(tokenize_function, batched=True)
    val = val.map(tokenize_function, batched=True)
    test = test.map(tokenize_function, batched=True)
    
    # Model
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
    )
    trainer.train()
    
    # Inference
    model.eval()
    generated = []
    for _ in range(5):  # Generate 5 samples
        input_ids = tokenizer.encode("{\"resourceType\": \"Bundle\"", return_tensors="pt")
        output = model.generate(input_ids, max_length=512, num_return_sequences=1, do_sample=True, top_p=0.9)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            bundle = json.loads(text)
            generated.append(bundle)
        except json.JSONDecodeError:
            continue
    
    # Output samples
    print("Generated Synthetic FHIR Bundles:")
    for i, bundle in enumerate(generated[:3]):  # Show first 3
        print(f"Sample {i+1}: {json.dumps(bundle, indent=2)}")

if __name__ == "__main__":
    main()
