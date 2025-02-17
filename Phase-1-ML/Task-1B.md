
## Task 1: NumPy Array Normalization  
Create a function `normalize_columns` that:  
- Accepts 2D NumPy array of numerical data  
- Returns new array where each column is normalized using z-score:  
  `(x - mean)/std_dev`  
- Handle columns with zero standard deviation by returning original values  

**Example Input:**  
`[[1.0, 2.0], [3.0, 4.0], [5.0, 2.0]]`  

---

## Task 2: Pandas Data Loading & Filtering  
1. Load CSV data into DataFrame using `pd.read_csv()`  
2. Remove rows with missing values in specified columns  
3. Create function `filter_by_range(df, col_name, min_val, max_val)`  
   that returns filtered DataFrame  
4. Calculate mean and standard deviation for numerical columns  

---

## Task 3: NumPy Array Reshaping and Slicing  
Implement:  
1. Create 3D array (10x28x28) representing 10 grayscale images  
2. Flatten to 2D array (10x784) using reshaping  
3. Create function `extract_patch(arr, patch_size=5)` that extracts  
   center patches from each image  
4. Normalize patches to 0-1 range  


## Task 4: Pandas Pivot Table Creation  
1. Load sales data with columns: ["Date", "Product", "Quantity", "Price"]  
2. Create pivot table showing total sales per product per month  
3. Add "Total_Sales" column (Quantity * Price)  
4. Handle missing values by filling with 0  
5. Sort results by Total_Sales descending  


## Task 5: Pandas Categorical Encoding  
1. Convert text columns to categorical using one-hot encoding  
2. Implement binning for numerical columns (e.g., age groups)  
3. Create function `preprocess_data(df)` that returns:  
   - DataFrame with dummy variables for categorical columns  
   - Binned numerical columns  
   - Scaled numerical columns (0-1 range)  

---
