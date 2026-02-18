from rich.console import Console
from rich.markdown import Markdown
import plotext as plt

console = Console()

# Generate a plotext graph
plt.clf()
plt.theme("dark")
plt.frame(True)
plt.grid(True)
plt.title("Test Graph")
plt.plot([1, 2, 3], [1, 4, 9], label="x^2")
graph_output = plt.build()

text = f"""
# Here is a graph
Some text before.

{graph_output}

Some text after.
"""

print("--- Printing using standard print ---")
print(text)

print("\n--- Printing using Rich Markdown ---")
# Rich Markdown might strip ANSI codes or mess up formatting.
try:
    console.print(Markdown(text))
except Exception as e:
    print(f"Rich Markdown failed: {e}")

print("\n--- Printing using Console.print (text) ---")
console.print(text)

print("\n--- Printing using Split Approach ---")
# Simulate splitting the text
parts = text.split(graph_output)
if len(parts) > 1:
    pre_text = parts[0]
    post_text = parts[1]

    console.print(Markdown(pre_text))
    print(graph_output)  # Print raw ansi
    console.print(Markdown(post_text))
else:
    console.print(Markdown(text))
