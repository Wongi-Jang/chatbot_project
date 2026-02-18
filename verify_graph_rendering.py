from scripts.rag.helper import render_text_with_graphs
from rich.console import Console
from rich.markdown import Markdown

console = Console()

text = """
Here is a graph:

```json:graph
{
    "type": "bar",
    "title": "Sales Data",
    "data": [
        {"label": "Q1", "x": ["Jan", "Feb", "Mar"], "y": [10, 15, 7]},
        {"label": "Q2", "x": ["Apr", "May", "Jun"], "y": [12, 18, 9]}
    ]
}
```

And some text after.
"""

print("--- Testing render_text_with_graphs ---")
segments = render_text_with_graphs(text)

for seg in segments:
    if seg["type"] == "text":
        console.print(Markdown(seg["content"]))
    else:
        print(seg["content"])

print("\n--- Done ---")
