# test methodology

Example programs and generated datasets for testing, fuzzing, and debugging purposes:

**I. Introduction**

* Brief overview of the importance of testing, fuzzing, and debugging in software development
* Purpose of this list: providing examples of programs and datasets for testing, fuzzing, and debugging

**II. Example Programs**

A. **Simple Arithmetic Operations**

* Addition, subtraction, multiplication, division (e.g., `add(int a, int b)`, `sub(int a, int b)` etc.)
* Examples:
	+ Input: 2, 3
	+ Expected output: 5

B. **String Manipulation**

* Concatenation, substring extraction, string comparison (e.g., `concatenate(String s1, String s2)`, `extractSubstring(String s, int start, int end)` etc.)
* Examples:
	+ Input: "hello", "world"
	+ Expected output: "helloworld"

C. **Data Structures**

* Arrays, lists, stacks, queues (e.g., `push(int val, Stack stack)`, `pop(Stack stack)` etc.)
* Examples:
	+ Input: [1, 2, 3]
	+ Expected output: []

D. **Network Communication**

* Socket programming, HTTP requests (e.g., `connect(String host, int port)`, `sendRequest(String url)` etc.)
* Examples:
	+ Input: "example.com", 80
	+ Expected output: "HTTP/1.1 200 OK"

**III. Generated Datasets**

A. **Random Numbers**

* Integers, floats, strings (e.g., `randomInt()`, `randomFloat()` etc.)
* Examples:
	+ Input: None
	+ Expected output: [10, 20, 30]

B. **Test Cases**

* Example inputs and expected outputs for each program (e.g., test cases for addition: [(2, 3), (4, 5)])
* Examples:
	+ Input: [(1, 2), (3, 4)]
	+ Expected output: [3, 7]

C. **Fuzz Test Cases**

* Malicious or invalid inputs to test error handling and robustness (e.g., NaNs, infinities, null values)
* Examples:
	+ Input: [(NaN, 2), (inf, 4)]
	+ Expected output: Error or Exception

D. **Edge Cases**

* Boundary testing for limits and constraints (e.g., max/min values, buffer overflows)
* Examples:
	+ Input: [1, 1000], [0, -10]
	+ Expected output: Errors or Exceptions for invalid inputs

