from setuptools import setup

setup(
    name="code2mermaid",
    version="0.1.0",
    description="Generate Mermaid flowcharts from Python code, escaping special characters to avoid parse errors.",
    py_modules=["code2mermaid"],
    entry_points={
        "console_scripts": [
            # This creates a CLI command "code2mermaid" that runs code2mermaid.py's main() function
            "code2mermaid=code2mermaid:main",
        ],
    },
    python_requires=">=3.6",
)

