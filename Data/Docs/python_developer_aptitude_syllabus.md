# Python Developer Aptitude Round Syllabus

## 1. Quantitative Aptitude

### Number System
- Natural, whole, integer, rational numbers
- Divisibility rules
- Prime numbers and factors
- HCF and LCM
- Remainders
- Unit digit and last-digit problems
- Powers and roots

### Arithmetic
- Percentage
- Profit and loss
- Simple interest
- Compound interest
- Ratio and proportion
- Average
- Mixture and allegation
- Partnership
- Age problems

### Time-Based Problems
- Time and work
- Pipes and cisterns
- Time, speed and distance
- Trains
- Boats and streams

### Data Interpretation
- Tables
- Bar charts
- Pie charts
- Line graphs
- Percentage-based data questions
- Average and ratio from datasets

### Basic Algebra
- Linear equations
- Quadratic equations
- Simplification
- Algebraic identities
- Sequences and series

---

## 2. Logical Reasoning

### Series
- Number series
- Alphabet series
- Alphanumeric series
- Missing-number problems

### Coding and Decoding
- Letter coding
- Number coding
- Pattern-based coding

### Arrangements
- Seating arrangement
- Linear arrangement
- Circular arrangement
- Ranking and ordering

### Logical Problems
- Blood relations
- Direction sense
- Syllogisms
- Statements and conclusions
- Statements and assumptions
- Cause and effect
- Puzzles

### Analytical Reasoning
- Data sufficiency
- Odd one out
- Analogy
- Venn diagrams
- Logical deduction
- Pattern recognition

---

## 3. Verbal Ability

### English Grammar
- Parts of speech
- Tenses
- Articles
- Prepositions
- Subject-verb agreement
- Active and passive voice
- Direct and indirect speech
- Conjunctions
- Sentence structure

### Vocabulary
- Synonyms
- Antonyms
- One-word substitutions
- Commonly confused words

### Reading and Writing
- Reading comprehension
- Sentence completion
- Error detection
- Sentence correction
- Para jumbles
- Fill in the blanks

> For a Python developer, English aptitude is usually not the main technical filter, but many company assessments include it.

---

# 4. Python Programming Aptitude

This is the most important section for a Python developer.

## Python Basics
- Variables
- Data types
- Type conversion
- Operators
- Input/output
- `if`, `elif`, `else`
- `for` and `while` loops
- `break`, `continue`, `pass`

## Python Data Structures
- List
- Tuple
- Set
- Dictionary
- String

Know:
- Creation
- Indexing
- Slicing
- Common methods
- Mutability vs immutability
- When to use each structure

### Important Methods

#### List
- `append()`
- `extend()`
- `insert()`
- `remove()`
- `pop()`
- `sort()`
- `reverse()`

#### String
- `split()`
- `join()`
- `replace()`
- `strip()`
- `find()`
- `startswith()`
- `endswith()`

#### Dictionary
- `keys()`
- `values()`
- `items()`
- `get()`
- `update()`
- `pop()`

#### Set
- Union
- Intersection
- Difference
- Symmetric difference

---

# 5. Python Output-Based Questions

Practice predicting the output without running the code.

Topics:
- Variable references
- Mutable vs immutable objects
- String slicing
- List slicing
- Nested lists
- `is` vs `==`
- Truthy and falsy values
- Operator precedence
- `*args` and `**kwargs`
- Default arguments
- Scope
- Loop behavior
- Exceptions

Example:

```python
a = [1, 2, 3]
b = a
b.append(4)

print(a)
```

You should understand why the output is:

```text
[1, 2, 3, 4]
```

---

# 6. Functions

Study:

- Function definition
- Parameters and arguments
- Positional arguments
- Keyword arguments
- Default arguments
- `*args`
- `**kwargs`
- Return values
- Scope
- Local and global variables
- Lambda functions
- Higher-order functions
- Recursion

Important built-ins:
- `map()`
- `filter()`
- `reduce()`
- `zip()`
- `enumerate()`
- `sorted()`
- `any()`
- `all()`

---

# 7. Object-Oriented Programming

Very commonly asked in Python interviews.

Topics:
- Class and object
- Constructor (`__init__`)
- Instance variables
- Class variables
- Instance methods
- Class methods
- Static methods
- Encapsulation
- Inheritance
- Polymorphism
- Abstraction
- Method overriding
- `super()`
- Magic/dunder methods

Important dunder methods:
- `__init__`
- `__str__`
- `__repr__`
- `__len__`
- `__eq__`

Be able to explain the difference between:

```text
Class
Object
Instance variable
Class variable
```

---

# 8. Exception Handling

Topics:
- `try`
- `except`
- `else`
- `finally`
- `raise`
- Custom exceptions
- Common Python exceptions

Know common exceptions:
- `TypeError`
- `ValueError`
- `KeyError`
- `IndexError`
- `AttributeError`
- `NameError`
- `ZeroDivisionError`
- `FileNotFoundError`

---

# 9. File Handling

Study:
- Opening files
- Reading files
- Writing files
- Append mode
- File modes
- `with open(...)`
- CSV basics
- JSON basics

Example:

```python
with open("data.txt", "r") as file:
    data = file.read()
```

Understand why `with` is preferred.

---

# 10. Python Advanced Concepts

For better-paying or experienced Python roles, prepare:

- List comprehensions
- Dictionary comprehensions
- Set comprehensions
- Generator expressions
- Iterators
- Generators
- `yield`
- Decorators
- Closures
- Context managers
- `__name__ == "__main__"`
- Shallow copy vs deep copy
- References
- Garbage collection
- Python memory management

---

# 11. Data Structures and Algorithms

You do not need competitive-programming-level DSA for every Python job, but basic DSA is extremely important.

## Data Structures
- Array/List
- String
- Stack
- Queue
- Linked list
- Hash table / Dictionary
- Set
- Tree basics
- Graph basics

## Algorithms
- Linear search
- Binary search
- Bubble sort
- Selection sort
- Insertion sort
- Merge sort basics
- Recursion
- Two-pointer technique
- Sliding window basics

## Complexity
Understand:

```text
O(1)
O(log n)
O(n)
O(n log n)
O(n²)
```

Be able to identify the approximate time complexity of simple Python code.

---

# 12. Coding Problems to Practice

Start with easy problems and gradually increase difficulty.

## Beginner
- Reverse a string
- Check palindrome
- Find largest number
- Find smallest number
- Count vowels
- Count character frequency
- Remove duplicates
- Find duplicate elements
- Sum of list elements
- Find second-largest number
- Check prime number
- Generate Fibonacci series
- Find factorial
- Reverse a number

## Intermediate
- Two Sum
- Anagram check
- First non-repeating character
- Merge two sorted lists
- Find missing number
- Find common elements
- Move zeros to the end
- Find maximum subarray
- Count word frequency
- Flatten a nested list
- Sort a dictionary
- Group anagrams

---

# 13. SQL Aptitude for Python Developers

If you apply for backend/Django roles, SQL is important.

## SQL Basics
- `SELECT`
- `WHERE`
- `ORDER BY`
- `GROUP BY`
- `HAVING`
- `DISTINCT`
- `LIMIT`

## SQL Operations
- `INSERT`
- `UPDATE`
- `DELETE`

## Joins
- INNER JOIN
- LEFT JOIN
- RIGHT JOIN
- FULL OUTER JOIN

## Other Important Topics
- Primary key
- Foreign key
- Unique key
- Constraints
- Indexes
- Aggregate functions
- Subqueries
- Views
- Transactions
- Normalization basics

Practice:

```sql
COUNT()
SUM()
AVG()
MIN()
MAX()
```

---

# 14. Django / Backend Aptitude

If the job description mentions Django, Flask, or FastAPI, prepare these separately.

## Django
- Django architecture
- MVT
- Models
- Views
- URLs
- Templates
- ORM
- QuerySets
- Migrations
- Admin
- Forms
- Middleware
- Authentication
- Permissions
- Sessions
- Static and media files

## Django REST Framework
- REST API concepts
- HTTP methods
- Status codes
- Serializers
- APIView
- Generic views
- ViewSets
- Routers
- Authentication
- Permissions
- Pagination
- Filtering
- Validation

## Backend Concepts
- REST API
- JSON
- HTTP/HTTPS
- Cookies
- Sessions
- JWT
- Authentication vs authorization
- CORS
- API status codes

---

# 15. Computer Science Fundamentals

Many companies include basic CS questions.

## Operating Systems
- Process vs thread
- Multithreading
- Multiprocessing
- Deadlock basics
- Memory management
- CPU scheduling basics

## Networking
- HTTP vs HTTPS
- TCP vs UDP
- IP address
- DNS
- Client-server architecture
- Request/response
- Common HTTP status codes

## Database
- SQL vs NoSQL
- Relational databases
- Transactions
- ACID
- Indexing
- Normalization

## Git
- `git clone`
- `git init`
- `git add`
- `git commit`
- `git push`
- `git pull`
- Branches
- Merge
- Rebase basics
- Merge conflicts

---

# 16. Aptitude Test Strategy

A typical Python developer hiring process may look like:

```text
Aptitude
   ↓
Python MCQs
   ↓
SQL / CS Fundamentals
   ↓
Coding Test
   ↓
Technical Interview
   ↓
HR / Managerial Round
```

The exact process varies by company.

## Priority Order

If your preparation time is limited, use this order:

### Priority 1
- Python basics
- Lists, dictionaries, sets, strings
- Functions
- OOP
- Output-based questions
- Basic coding problems

### Priority 2
- DSA
- SQL
- Exception handling
- Iterators/generators
- Decorators
- Complexity

### Priority 3
- Django/DRF
- HTTP/REST
- Git
- OS
- Networking
- Database concepts

### Priority 4
- Quantitative aptitude
- Logical reasoning
- Verbal ability

---

# 17. 30-Day Preparation Plan

## Week 1: Python Fundamentals

**Day 1**
- Variables
- Data types
- Operators

**Day 2**
- Conditions
- Loops

**Day 3**
- Strings

**Day 4**
- Lists and tuples

**Day 5**
- Sets and dictionaries

**Day 6**
- Functions
- `*args`
- `**kwargs`

**Day 7**
- Python MCQs + output questions

---

## Week 2: Core Python + OOP

**Day 8**
- Scope
- Lambda
- `map`
- `filter`
- `reduce`

**Day 9**
- Comprehensions

**Day 10**
- Exception handling
- File handling

**Day 11**
- Classes and objects

**Day 12**
- Inheritance
- Polymorphism
- Encapsulation

**Day 13**
- Magic methods
- Class/static methods

**Day 14**
- Python mock test

---

## Week 3: DSA + SQL

**Day 15**
- Big-O
- Arrays/lists
- Strings

**Day 16**
- Stack
- Queue
- Hashing

**Day 17**
- Searching
- Sorting

**Day 18**
- Recursion
- Two pointers

**Day 19**
- SQL basics

**Day 20**
- Joins
- Grouping
- Aggregations

**Day 21**
- SQL + DSA mock test

---

## Week 4: Backend + Interview Preparation

**Day 22**
- Django fundamentals

**Day 23**
- Django ORM

**Day 24**
- Django REST Framework

**Day 25**
- REST API
- HTTP
- Authentication

**Day 26**
- Git
- Linux basics

**Day 27**
- OS
- Networking
- Database fundamentals

**Day 28**
- Python coding test

**Day 29**
- Full aptitude + technical mock test

**Day 30**
- Review weak topics
- Practice interview questions

---

# 18. Target for Job Preparation

Before applying, aim to be comfortable with:

- [ ] 100+ Python MCQs
- [ ] 50+ Python output questions
- [ ] 30+ coding problems
- [ ] 50+ SQL questions
- [ ] 20+ DSA problems
- [ ] 20+ Django questions
- [ ] 20+ REST API questions
- [ ] 10+ OOP interview questions
- [ ] 5+ full mock tests

## Most Important Rule

Do not prepare only by reading.

Use this cycle:

```text
Learn
  ↓
Write code
  ↓
Solve MCQs
  ↓
Solve without IDE/autocomplete
  ↓
Review mistakes
  ↓
Repeat
```

For a Python developer, **being able to predict code behavior and write small programs from scratch is more valuable than memorizing Python definitions.**
