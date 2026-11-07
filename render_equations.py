import subprocess, os
eqs = {
1: r"M_v(\Delta)=\frac{\mathrm{SID}}{\mathrm{SOD}_0+\Delta},\qquad M_h(\Delta)=M_0\,\frac{x_0}{x_0+\Delta},\qquad \mathrm{SOD}_0=\frac{\mathrm{SID}}{M_0}",
2: r"h=1+s=\frac{M_h(\Delta)}{M_0}\quad\Longrightarrow\quad \Delta=-\,\frac{s\,x_0}{1+s}",
3: r"\theta=\arctan\!\left(\frac{k\,H}{d_{\mathrm{ap}}}\right),\qquad \varphi=\arctan\!\left(\frac{t\,H}{L}\right)",
4: r"\delta_{i}(u)=\hat{y}\big(\mathcal{T}_{u}(I_i)\big)-\hat{y}\big(\mathcal{T}_{0}(I_i)\big)",
5: r"\sigma_0=\sqrt{\frac{1}{N(R-1)}\sum_{i=1}^{N}\sum_{j=1}^{R}\big(\hat{y}_{ij}-\bar{y}_{i}\big)^{2}}",
6: r"\delta_{i}(u)=a+b\,u+\varepsilon_{i},\qquad \varepsilon_i\sim\mathcal{N}(0,\sigma^2)",
7: r"\mathrm{LoD}=\frac{\big(z_{1-\alpha}+z_{1-\beta}\big)\,\sigma_0}{b}=\frac{3.29\,\sigma_0}{b}\qquad(\alpha=\beta=0.05)",
8: r"d_i=\hat{y}_i-\operatorname*{med}_{j=1,\dots,N}\hat{y}_j,\qquad \text{patient } i \text{ above LoD} \iff |d_i|>\mathrm{LoD}",
}
tpl=r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{mathptmx}
\begin{document}
\begin{minipage}{15.5cm}
\begin{equation*}
%s \tag{%d}
\end{equation*}
\end{minipage}
\end{document}
"""
for n,e in eqs.items():
    f=f"eq{n}"
    open(f+".tex","w").write(tpl%(e,n))
    r=subprocess.run(["pdflatex","-interaction=nonstopmode",f+".tex"],capture_output=True,text=True)
    if r.returncode: print(n, r.stdout[-800:]); continue
    subprocess.run(["pdftoppm","-png","-r","250","-singlefile",f+".pdf",f"Equation{n}"])
print(sorted(os.listdir('.')))
