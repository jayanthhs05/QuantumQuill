## Task 1: List-to-Dictionary Converter  
Create a function `list_to_dict` that takes two lists:  
- `keys` (list of strings)  
- `values` (list of integers)  

Return a dictionary where each key-value pair is formed by combining elements at the same index from both lists. If the lists have unequal lengths, use `None` for missing values.  

**Example Input:**  
`keys = ["a", "b", "c"]`, `values = [1, 2]`  
**Expected Output:**  
`{"a": 1, "b": 2, "c": None}`  

---

## Task 2: Inventory Management System
Design a `Product` class with:  
- Attributes: `name` (str), `price` (float), `quantity` (int)  
- Method: `update_quantity(new_quantity)` to modify stock  
- Method: `get_total_value()` returning `price * quantity`  

Create a function `get_low_stock_products(products, threshold)` that accepts a list of `Product` objects and returns product names with `quantity` below the threshold.  

---

## Task 3: Student Grade Analyzer  
Write a program to:  
1. Create a dictionary `student_grades` where keys are student names (str) and values are lists of integers (e.g., `{"Jayanth": [85, 90], "Atharva": [100, 100]}`).  
2. Add a function `calculate_averages()` that returns a new dictionary with student names and their average grades.  
3. Add a function `find_top_student()` to identify the student with the highest average.  

---

## Task 4: Dictionary Merge with Conflict Resolution  
Write a function `merge_dicts(dict1, dict2)` that:  
- Combines two dictionaries  
- If keys conflict, sums the integer values  
- Returns the merged dictionary  

**Example Input:**  
`dict1 = {"a": 5, "b": 10}`, `dict2 = {"a": 3, "c": 7}`  
**Expected Output:**  
`{"a": 8, "b": 10, "c": 7}`  

---

## Task 5: Library System
Implement a `Library` class with:  
- Attribute: `catalog` (list of dictionaries, where each dict has `title`, `author`, `is_available` keys)  
- Method: `borrow_book(title)` to mark a book as unavailable  
- Method: `get_available_books()` returning titles of available books  
- Method: `add_book(title, author)` to extend the catalog  

Include error handling for invalid operations like borrowing unavailable books.  

--- 
