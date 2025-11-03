import React from "react";

export default function Badges() {
  return (
    <p style={{display: "flex", gap: "8px", flexWrap: "wrap"}}>
      <a href="https://arxiv.org/abs/2505.23719" target="_blank" rel="noreferrer">
        <img
          alt="arXiv"
          src="https://img.shields.io/static/v1?label=Paper&message=2505.23719&color=B31B1B&logo=arXiv"
          height="22"
        />
      </a>
      <a href="https://github.com/NX-AI/tirex" target="_blank" rel="noreferrer">
        <img
          alt="GitHub"
          src="https://img.shields.io/badge/GitHub-NX--AI%2Ftirex-181717?logo=github"
          height="22"
        />
      </a>
      <a href="https://huggingface.co/NX-AI/TiRex" target="_blank" rel="noreferrer">
        <img
          alt="Hugging Face"
          src="https://img.shields.io/badge/HuggingFace-TiRex-yellow?logo=huggingface"
          height="22"
        />
      </a>
      <a href="https://pypi.org/project/tirex-ts/" target="_blank" rel="noreferrer">
        <img
          alt="PyPI"
          src="https://img.shields.io/pypi/v/tirex-ts?color=0a84ff"
          height="22"
        />
      </a>
    </p>
  );
}
