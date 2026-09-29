import scratch.yeganeh_polynomial_halftone as yp

# We will modify the function slightly to just return the AST string instead of rendering.
def get_ast():
    import numpy as np
    from PIL import Image
    
    img = Image.open("/Users/babayaga/image to equation/einstein_matched.jpg").convert('L')
    res = 100
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    target = np.array(img).astype(np.float32) / 255.0
    
    y_lin = np.linspace(-1, 1, res)
    x_lin = np.linspace(-1, 1, res)
    Y, X = np.meshgrid(y_lin, x_lin, indexing='ij')
    
    X_f = X.flatten()
    Y_f = Y.flatten()
    T_f = target.flatten()
    
    D = 25
    terms_list = []
    for d in range(D + 1):
        for i in range(d + 1):
            j = d - i
            terms_list.append((i, j))
            
    A = np.zeros((len(X_f), len(terms_list)))
    for idx, (p_x, p_y) in enumerate(terms_list):
        A[:, idx] = (X_f ** p_x) * (Y_f ** p_y)
        
    coeffs, _, _, _ = np.linalg.lstsq(A, T_f, rcond=None)
    
    ast_terms = []
    for idx, (p_x, p_y) in enumerate(terms_list):
        c = coeffs[idx]
        if abs(c) < 1e-4: continue
        term = f"{c:.5f}"
        if p_x > 0:
            if p_x == 1: term += "*x"
            else: term += f"*(x^{p_x})"
        if p_y > 0:
            if p_y == 1: term += "*y"
            else: term += f"*(y^{p_y})"
        ast_terms.append(term)
        
    poly_expr = " + ".join(ast_terms)
    
    full_ast = f"""let(x, x*2-1, let(y, y*2-1, let(brightness, clamp({poly_expr}, 0.0, 1.0), let(pattern, cos(200 * sqrt(x^2 + y^2)), sdfmask(pattern - (brightness*2-1), 0.1)))))"""
    return full_ast

if __name__ == "__main__":
    ast_str = get_ast()
    with open("full_einstein_ast.txt", "w") as f:
        f.write(ast_str)
