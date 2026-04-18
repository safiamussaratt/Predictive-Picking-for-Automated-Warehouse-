# Predictive Picking for Automated Warehouse
This project combines Machine Learning and Informed Search Algorithms to optimize warehouse picking operations. Using the Instacart Market Basket Analysis dataset, we predict items likely to be purchased together and compute optimal picking paths using the A* algorithm.

### Task 1: Machine Learning - Association Prediction

#### Dataset
- **Source:** [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
- **Size:** 10,000 orders sampled, 300 most frequent products
- **Features:** Product ID pairs (one-hot encoded)
- **Labels:** Binary (1 = co-purchased, 0 = not co-purchased)

#### Data Preprocessing Pipeline

```python
Raw Orders → Filter Top Products → Generate Item Pairs → Balance Classes → One-Hot Encode
```

| Step | Description | Output Shape |
|------|-------------|--------------|
| 1. Filter | Keep top 300 products by frequency | 10,000 orders |
| 2. Pair Generation | Create (A,B) pairs from each order | ~50,000 positive pairs |
| 3. Class Balancing | Sample equal negative pairs | ~100,000 total samples |
| 4. Encoding | One-hot encode both products | (n_samples, 600) |

#### Models Implemented

| Model | Description | Advantages | Disadvantages |
|-------|-------------|------------|---------------|
| **Logistic Regression** | Linear classifier with L2 regularization | Fast training, interpretable | Cannot capture non-linear patterns |
| **Random Forest** | Ensemble of 100 decision trees | Handles non-linearity, robust | Slower training, more memory |

#### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | How many predicted associations are correct |
| **Recall** | TP/(TP+FN) | How many actual associations we found |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean - best for imbalanced data |

### Task 2: Informed Search - A* Algorithm

#### Algorithm Selection Justification

| Criteria | A* Performance |
|----------|----------------|
| **Optimality** | Guaranteed with admissible heuristic |
| **Completeness** | Always finds path if one exists |
| **Efficiency** | Prunes search space with heuristic |
| **Grid Suitability** | Naturally handles 2D grid navigation |

#### Implementation Details

```python
f(n) = g(n) + h(n)

where:
  g(n) = actual cost from start to node n (steps taken)
  h(n) = Manhattan distance from n to goal
        = |x₁ - x₂| + |y₁ - y₂|
```

**Heuristic Properties:**
- **Admissible:** Never overestimates true cost (Manhattan ≤ actual path)
- **Consistent:** Satisfies triangle inequality h(n) ≤ c(n,n') + h(n')

#### Multi-Goal Path Planning Strategy

```python
Algorithm: Greedy Sequential A*

1. Start at (0,0)
2. While goals remain:
   a. Run A* from current position to each remaining goal
   b. Select nearest reachable goal
   c. Move to that goal and remove from set
3. Return complete path and total distance
```

---

## Installation & Usage

### Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| Python | ≥ 3.8 | [python.org](https://python.org) |
| pip | Latest | Included with Python |
| Git | Any | [git-scm.com](https://git-scm.com) |

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Predictive-Picking-Automated-Warehouse.git
cd Predictive-Picking-Automated-Warehouse
```

#### 2️. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `matplotlib` - Visualization
- `scipy` - Sparse matrix support

#### 3️. Download the Dataset

1. Visit [Instacart Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
2. Click **Download** (requires Kaggle account)
3. Extract all CSV files to the `data/` folder

**Expected Files:**
```
data/
├── orders.csv
├── order_products__prior.csv
├── products.csv
├── aisles.csv
└── departments.csv
```

#### 4️. Configure Parameters (Optional)

Edit these variables in `predictive_picking.py`:

```python
# Configuration Section
DATASET_PATH = "data/"          # Path to dataset folder
SAMPLE_ORDERS = 10000           # Orders to process (reduce if memory issues)
TOP_N_PRODUCTS = 300            # Most frequent products to consider
GRID_SIZE = 10                  # Warehouse grid dimensions
OBSTACLE_DENSITY = 0.25         # Percentage of grid as shelves
TOP_K_ITEMS = 3                 # Number of items to predict per order
RANDOM_SEED = 42                # For reproducibility
```

#### 5️. Run the Project

```bash
python predictive_picking.py
```


### Visual Results

| Optimized Route | Baseline Random Route |
|-----------------|----------------------|
| Intelligent goal ordering minimizes backtracking | Random ordering causes inefficient criss-crossing |
| A* finds direct Manhattan paths | Same A* but suboptimal goal sequence |
| Natural picking flow | Unnecessary travel between distant points |

---

## Project Structure

```
Predictive-Picking-Automated-Warehouse/
│
├── project.py         # Main implementation (600+ lines)
├── README.md                     # This documentation

│
├── data/                         # Dataset (not tracked in Git)
│   ├── orders.csv                   # Order metadata
│   ├── order_products__prior.csv    # Product-order mappings
│   ├── products.csv                 # Product catalog
│   ├── aisles.csv                   # Aisle information
│   └── departments.csv              # Department information
│
├── images/                       # Generated visualizations
│   ├── optimized_route.png          # Optimized path plot
│   └── baseline_route.png           # Baseline path plot
