import streamlit as st
import math
from typing import Union, Tuple

def solve_quadratic_equation(a: float, b: float, c: float) -> Tuple[str, Union[Tuple[float, float], None]]:
    """
    Giải phương trình bậc 2: ax² + bx + c = 0
    
    Args:
        a (float): Hệ số của x²
        b (float): Hệ số của x
        c (float): Hệ số tự do
        
    Returns:
        Tuple[str, Union[Tuple[float, float], None]]: Kết quả giải phương trình và nghiệm (nếu có)
    """
    # Trường hợp phương trình bậc 1 (a = 0)
    if a == 0:
        if b == 0:
            return 'Phương trình có vô số nghiệm', None
        x = -c / b
        return f'Phương trình có nghiệm x = {x:.2f}', (x, None)
    
    # Trường hợp phương trình bậc 2 (a ≠ 0)
    delta = b**2 - 4*a*c
    if delta < 0:
        return 'Phương trình vô nghiệm', None
    
    x1 = (-b + math.sqrt(delta)) / (2*a)
    x2 = (-b - math.sqrt(delta)) / (2*a)
    return f'Phương trình có hai nghiệm: x₁ = {x1:.2f} và x₂ = {x2:.2f}', (x1, x2)

def clear_inputs() -> None:
    """Xóa tất cả các giá trị nhập vào"""
    st.session_state["input_a"] = 0.0
    st.session_state["input_b"] = 0.0
    st.session_state["input_c"] = 0.0

def main() -> None:
    """Hàm chính của ứng dụng"""
    st.title('Giải Phương Trình Bậc 2')
    st.markdown('---')
    
    with st.form(key='equation_form', clear_on_submit=False):
        st.subheader('Nhập các hệ số')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a = st.number_input('Hệ số a', key='input_a', format='%.2f')
        with col2:
            b = st.number_input('Hệ số b', key='input_b', format='%.2f')
        with col3:
            c = st.number_input('Hệ số c', key='input_c', format='%.2f')
        
        col1, col2 = st.columns(2)
        with col1:
            solve_button = st.form_submit_button('Giải Phương Trình')
        with col2:
            clear_button = st.form_submit_button('Xóa', on_click=clear_inputs)
        
        if solve_button:
            result, _ = solve_quadratic_equation(a, b, c)
            st.markdown('### Kết quả:')
            st.success(result)
        else:
            st.markdown('### Kết quả:')
            st.info('Vui lòng nhập các hệ số và nhấn "Giải Phương Trình"')

if __name__ == '__main__':
    main()
