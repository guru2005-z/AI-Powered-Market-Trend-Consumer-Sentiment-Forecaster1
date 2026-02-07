# Sentiment Analysis Model Validation Guide

## Overview
The `validate.py` script compares AI predictions against human labels to measure model accuracy.

---

## Key Concepts Explained

### 1. **df.sample(n=20, random_state=42)**

**What it does:**
- Randomly selects 20 rows from the DataFrame
- `n=20` specifies the number of rows to select
- `random_state=42` is the **"seed"** for randomness

**Why random_state is important:**
```python
# Without random_state:
df.sample(n=20)  # Different rows picked each run ❌

# With random_state:
df.sample(n=20, random_state=42)  # Same rows ALWAYS picked ✅
```

**Benefits:**
- **Reproducibility**: Same results every run (great for testing)
- **Consistency**: You can validate the exact same samples
- **Debugging**: Easier to track issues with consistent data
- **Reporting**: Others can replicate your results

**Common random_state values:**
- `random_state=42` - Most popular (convention in ML)
- `random_state=0` - Common alternative
- `random_state=None` - Truly random (no reproducibility)

---

### 2. **accuracy_score(human_labels, ai_labels)**

**Formula:**
```
Accuracy = (Correct Predictions) / (Total Predictions) × 100%
```

**Example:**
```
Human:  [positive, negative, positive, neutral, positive]
AI:     [positive, negative, positive, positive, positive]
                    ✓        ✓        ✓        ✗        ✓
Correct predictions: 4 out of 5
Accuracy: (4/5) × 100 = 80%
```

**Interpretation:**
| Accuracy Range | Performance |
|---|---|
| 90-100% | Excellent ⭐⭐⭐⭐⭐ |
| 80-90% | Good ⭐⭐⭐⭐ |
| 70-80% | Fair ⭐⭐⭐ |
| 60-70% | Poor ⭐⭐ |
| <60% | Very Poor ⭐ |

---

### 3. **Precision & Recall (Classification Report)**

#### **Precision: "Are my predictions trustworthy?"**
```
Formula: TP / (TP + FP)

TP = True Positives (correctly predicted positive)
FP = False Positives (predicted positive but was negative)
```

**Example:**
- Model predicts 100 reviews as "Positive"
- 90 were actually positive (TP)
- 10 were actually negative (FP)
- **Precision = 90 / (90 + 10) = 90%**

**Interpretation:** 90% of the time, when the model says "Positive", it's correct.

**Use case:** When false positives are costly
- Email spam detection: Don't want to flag legitimate emails
- Fraud detection: Don't want to block real transactions

---

#### **Recall: "Am I catching all cases?"**
```
Formula: TP / (TP + FN)

TP = True Positives (correctly predicted positive)
FN = False Negatives (predicted negative but was positive)
```

**Example:**
- 100 reviews are actually "Positive"
- Model correctly identified 90 (TP)
- Model missed 10 (FN)
- **Recall = 90 / (90 + 10) = 90%**

**Interpretation:** 90% of all actual positive reviews are caught by the model.

**Use case:** When false negatives are costly
- Disease detection: Don't want to miss sick patients
- Security: Don't want to miss threats

---

#### **F1-Score: "Overall balance?"**
```
Formula: 2 × (Precision × Recall) / (Precision + Recall)
```

- **Harmonic mean** of Precision and Recall
- Balances both metrics
- Range: 0 (worst) to 1 (best)
- Use when you want a single overall score

**Example:**
- Precision = 0.90
- Recall = 0.88
- F1 = 2 × (0.90 × 0.88) / (0.90 + 0.88) = 0.89

---

## Improvements Made to Your Script

### ✅ **1. Case-Insensitive Input Handling**

**Before:**
```python
user_input = input("Your Label (p/n/u): ").strip().lower()
mapping = {'p': 'Positive', 'n': 'Negative', 'u': 'Neutral'}
human_label = mapping.get(user_input, 'Neutral')  # Defaults to Neutral if invalid
```

**Problems:**
- Silently defaults to "Neutral" on typo
- Only accepts single letters

**After:**
```python
valid_input = False
while not valid_input:
    user_input = input("Your Label (p/n/u or positive/negative/neutral): ").strip().lower()
    
    # Accept multiple formats
    mapping = {
        'p': 'positive', 'positive': 'positive',
        'n': 'negative', 'negative': 'negative',
        'u': 'neutral', 'neutral': 'neutral'
    }
    
    if user_input in mapping:
        human_label = mapping[user_input]
        valid_input = True
    else:
        print("❌ Invalid input! Use: p/n/u or positive/negative/neutral")
```

**Benefits:**
- Accepts: `p`, `P`, `positive`, `Positive`, `POSITIVE` ✅
- Rejects invalid input with error message ✅
- Prevents silent defaults ✅
- No data corruption from typos ✅

---

### ✅ **2. Mismatch Tracking**

**Added:**
```python
mismatches = []  # Track disagreements

# ... inside loop:
if human_label != ai_pred:
    mismatches.append({
        'review': display_text,
        'human': human_label,
        'ai': ai_pred
    })

# ... in report:
if mismatches:
    print(f"\n⚠️  Found {len(mismatches)} mismatch(es):")
    for i, m in enumerate(mismatches, 1):
        print(f"   {i}. Human: {m['human']} | AI: {m['ai']}")
```

**Benefits:**
- Identifies specific problem areas
- Helps debug model failures
- Useful for retraining

---

### ✅ **3. Confusion Matrix**

**Added:**
```python
cm = confusion_matrix(human_labels, ai_labels, 
                      labels=['positive', 'negative', 'neutral'])
cm_df = pd.DataFrame(cm, 
                    index=['Actual: Positive', 'Actual: Negative', 'Actual: Neutral'],
                    columns=['Predicted: Positive', 'Predicted: Negative', 'Predicted: Neutral'])
print(cm_df)
```

**Shows:**
```
                      Predicted: Positive  Predicted: Negative  Predicted: Neutral
Actual: Positive                       15                     2                    0
Actual: Negative                        1                    12                    2
Actual: Neutral                         0                     3                   10
```

**Interpretation:**
- Diagonal = Correct predictions ✅
- Off-diagonal = Errors ❌
- Tells you which classes are confused with each other

---

### ✅ **4. Input Validation**

**Added:**
```python
# Check if required columns exist
if 'content' not in df.columns or 'sentiment' not in df.columns:
    print("❌ Error: CSV must have 'content' and 'sentiment' columns")
    return

# Handle fewer than 20 reviews
if len(df_clean) < 20:
    print(f"⚠️  Only {len(df_clean)} reviews available. Need at least 20.")
    n_samples = len(df_clean)
```

**Benefits:**
- Clear error messages
- Handles edge cases gracefully
- No silent failures

---

### ✅ **5. Improved Output**

**Now shows:**
1. Explanation of key concepts
2. Review number out of total: `[5/20]`
3. Mismatch details with actual review text
4. Confusion matrix for detailed error analysis
5. Interpretation guide for understanding metrics

---

## How to Use

### **Run the validation:**
```bash
python src/analysis/validate.py
```

### **Expected input prompts:**
```
[1/20] Review: "Battery drains too fast..."
      AI Prediction: [negative]
      Your Label (p/n/u or positive/negative/neutral): n
      ✅ Match!
```

### **Output example:**
```
✓ Final Accuracy: 85.00%
✓ Correct Predictions: 17/20

📈 DETAILED METRICS:
              precision    recall  f1-score   support
    positive       0.88      0.90      0.89         7
    negative       0.83      0.80      0.81         5
     neutral       0.85      0.85      0.85         8
    accuracy                         0.85        20
```

---

## Common Issues & Solutions

### **Issue 1: "File not found"**
- Ensure `data/processed/youtube_sentiment.csv` exists
- Check file path is correct
- Verify CSV has `content` and `sentiment` columns

### **Issue 2: Low accuracy**
- Model may need retraining
- Check if data quality is good
- Review mismatches to identify patterns

### **Issue 3: High Precision, Low Recall**
- Model is conservative (avoids false positives)
- Missing some true cases
- Add more training examples of underrepresented class

---

## Summary Table

| Concept | Formula | Use Case |
|---|---|---|
| **Accuracy** | TP+TN / All | Overall performance |
| **Precision** | TP / (TP+FP) | When false positives costly |
| **Recall** | TP / (TP+FN) | When false negatives costly |
| **F1-Score** | 2×P×R/(P+R) | Balanced metric |

---

## Next Steps

1. ✅ Run validation on 20 random reviews
2. 📊 Analyze mismatches
3. 🔄 Retrain model if accuracy < 80%
4. 📈 Track accuracy over time as model improves
