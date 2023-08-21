[![Tests](https://github.com/Ruphoria/Ruphoria-Legal-Segmenter/actions/workflows/tests.yml/badge.svg)](https://github.com/Ruphoria/Ruphoria-Legal-Segmenter/actions/workflows/tests.yml)\n[![Documentation Status](https://readthedocs.org/projects/Ruphoria-Legal-Segmenter/badge/?version=latest)](https://Ruphoria-Legal-Segmenter.readthedocs.io/en/latest/?badge=latest)\n\n# Ruphoria Legal Text Segmenter\nThis project is a well-optimized Legal Text Segmenter for the Portuguese-Brazilian language.\n\nThe segmentation problem is formalized here using a 4-multiclass token-wise classification problem. Its algorithm can classify each token in the Portuguese legislative texts as follows:\n\n|Class |Description             |\n| :--- | :---                   |\n|0     |No-op                   |\n|1     |Start of sentence       |\n|2     |Start of noise sequence |\n|3     |End of noise sequence   |\n\n\nIn a curated dataset, comprised of ground-truth legal text segments, this Segmenter project achieves higher Precision and Recall for the Class 1 (Segment) than other segmentation tools. The repo also includes installation steps and usage examples for both BERTSegmenter and LSTMSegmenter.\n\nIf you plan to use optimized models in ONNX format, install the necessary optional dependencies. Tests for this package are run using Tox and Pytest.\n\n## License\n\n```markdown\nMIT License\n\nCopyright (c) 2022 Ruphoria\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \("Software"\)), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \("AS IS\"), WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n\n```