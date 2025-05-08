import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                return "Phương trình có vô số nghiệm"
            else:
                return "Phương trình vô nghiệm"
        else:
            x = -c/b
            return f"Phương trình có nghiệm x = {x:.2f}"
    else:
        delta = b**2 - 4*a*c
        if delta < 0:
            return "Phương trình vô nghiệm"
        elif delta == 0:
            x = -b/(2*a)
            return f"Phương trình có nghiệm kép x = {x:.2f}"
        else:
            x1 = (-b + np.sqrt(delta))/(2*a)
            x2 = (-b - np.sqrt(delta))/(2*a)
            return f"Phương trình có 2 nghiệm phân biệt:\nx₁ = {x1:.2f}\nx₂ = {x2:.2f}"

def plot_quadratic(a, b, c):
    # Create x values
    x = np.linspace(-10, 10, 1000)
    # Calculate y values
    y = a*x**2 + b*x + c
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', label=f'y = {a}x² + {b}x + {c}')
    
    # Add x and y axes
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Đồ thị hàm số bậc 2')
    
    # Add legend
    ax.legend()
    
    # Find and plot roots if they exist
    if a != 0:
        delta = b**2 - 4*a*c
        if delta >= 0:
            x1 = (-b + np.sqrt(delta))/(2*a)
            x2 = (-b - np.sqrt(delta))/(2*a)
            y1 = a*x1**2 + b*x1 + c
            y2 = a*x2**2 + b*x2 + c
            ax.plot([x1, x2], [y1, y2], 'ro', label='Nghiệm')
            ax.legend()
    
    return fig

def main():
    st.title("Giải phương trình bậc 2 và vẽ đồ thị")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nhập hệ số")
        a = st.number_input("Hệ số a:", value=1.0, step=0.1)
        b = st.number_input("Hệ số b:", value=0.0, step=0.1)
        c = st.number_input("Hệ số c:", value=0.0, step=0.1)

        if st.button("Giải phương trình"):
            result = solve_quadratic(a, b, c)
            st.subheader("Kết quả:")
            st.write(result)

    with col2:
        st.subheader("Đồ thị hàm số")
        fig = plot_quadratic(a, b, c)
        st.pyplot(fig)

if __name__ == "__main__":
    main() 