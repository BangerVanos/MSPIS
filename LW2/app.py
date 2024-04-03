# Методы решения задач в интеллектуальных системах
# Лабораторная работа №2 Вариант 7
# Авторы: Заломов Р.А., Готин И.А.
# Дата: 03.04.24
# Данный файл содержит графическую оболочку системы


from src.matrix_calculations import MatrixPU
from random import randint
import streamlit as st
import pandas as pd


class MatrixAppView:

    def __init__(self) -> None:
        st.set_page_config('Matrix app',
                           layout='wide')
    
    def run(self) -> None:
        with st.sidebar:
            st.number_input(label='P value',
                            min_value=1,
                            value=2,
                            key='p_value')
            st.number_input(label='M value',
                            min_value=1,
                            value=2,
                            key='m_value')
            st.number_input(label='Q value',
                            min_value=1,
                            value=2,
                            key='q_value')
            st.divider()
            st.number_input(label='Addition time',
                            min_value=1,
                            value=1,
                            key='add_time')
            st.number_input(label='Subtraction time',
                            min_value=1,
                            value=1,
                            key='sub_time')
            st.number_input(label='Multiply time',
                            min_value=1,
                            value=1,
                            key='mul_time')
            st.number_input(label='Divide time',
                            min_value=1,
                            value=1,
                            key='div_time')
            st.number_input(label='Compare time',
                            min_value=1,
                            value=1,
                            key='cpr_time')
            st.divider()
            st.number_input(label='Process elements',
                            min_value=1,
                            value=1,
                            key='procs_elems')
            st.number_input(label='Maximal processed vector size',
                            min_value=1,
                            value=1,
                            key='max_vec_size')
            st.button(label='Compute matrices',
                      type='primary',
                      on_click=self._compute_matrices)
    
    def _compute_matrices(self) -> None:        
        matrixpu = MatrixPU({
                     'p': st.session_state.get('p_value', 2),
                     'm': st.session_state.get('m_value', 2),
                     'q': st.session_state.get('q_value', 2),
                     'ADD_TIME': st.session_state.get('add_time', 1),
                     'SUB_TIME': st.session_state.get('sub_time', 1),
                     'MUL_TIME': st.session_state.get('mul_time', 1),
                     'DIV_TIME': st.session_state.get('div_time', 1),
                     'CPR_TIME': st.session_state.get('cpr_time', 1),
                     'PROCS_ELEMS': st.session_state.get('procs_elems', 1),
                     'MAX_VEC_SIZE': st.session_state.get('max_vec_size', 4)
                     })
        report = matrixpu.report        
        mat_col, stats_col = st.columns([10, 2])
        with mat_col:
            st.write('### Matrix A')
            st.table(pd.DataFrame(report['matrix_A']))
            st.write('### Matrix B')
            st.table(pd.DataFrame(report['matrix_B']))
            st.write('### Matrix E')
            st.table(pd.DataFrame([report['matrix_E']]))
            st.write('### Matrix G')
            st.table(pd.DataFrame(report['matrix_G']))
            st.write('### Matrix C (Result matrix)')
            st.table(pd.DataFrame(report['matrix_C']))
            st.write('### Matrix F')
            st.table(pd.DataFrame(report['matrix_F']))
        with stats_col:
            st.metric(label='Tacts of Sequential architecture',
                      value=report['seq_tacts'])
            st.metric(label='Tacts of Parallel architecture',
                      value=report['par_tacts'],
                      delta=(report['par_tacts'] - report['seq_tacts']),
                      delta_color='inverse')
            st.metric(label='Acceleration coefficent',
                      value=round(report['acceleration_coeff'], 3))
            st.metric(label='Efficency',
                      value=round(report['efficency'], 3))
            st.divider()
            st.metric(label='Used ADD (+) operations',
                      value=report['used_add'])
            st.metric(label='Used SUB (-) operations',
                      value=report['used_sub'])
            st.metric(label='Used MUL (*) operations',
                      value=report['used_mul'])
            st.metric(label='Used DIV (/) operations',
                      value=report['used_div'])
            st.metric(label='Used CPR (<=>) operations',
                      value=report['used_cpr'])
            st.divider()
            st.metric(label='Task Rank',
                      value=report['rank'])
            st.metric(label='Program Length (L)',
                      value=report['length'])
            st.metric(label='Average Program Length (Lavg)',
                      value=round(report['length_avg'], 3))
            st.metric(label='Divergence coefficient',
                      value=round(report['divergence_coeff'], 3))
            

view = MatrixAppView()
view.run()
