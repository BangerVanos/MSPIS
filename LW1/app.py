# Метода решения задач в интеллектуальных системах
# Лабораторная работа №1 Вариант 7
# Авторы: Заломов Р.А., Готин И.А.
# Дата: 28.02.24
# Данный файл содержит реализацию графической оболочки системы 

import streamlit as st
from src.arithmetic_pipeline import ArithmeticPipelineToJSON
import os
import json


class MainView:

    def __init__(self) -> None:
        st.set_page_config(page_title='Arithmetic pipeline')
        st.write('### Arithmetic pipeline')
        self._placeholder = st.empty()
        if not st.session_state.get('app_status'):
            st.session_state.app_status = 'render_parameters'
        if st.session_state.get('rendered_tact') is None:
            st.session_state.rendered_tact = 0

    def run(self):
        if st.session_state.get('app_status') == 'render_parameters':
            self._render_pipeline_parameters()
        elif st.session_state.get('app_status') == 'render_tacts':
            self._render_pipeline_tacts() 

    def _render_pipeline_parameters(self):
        with self._placeholder.container(border=False):
            st.write('Pipeline stages amount')
            st.number_input(label='Enter stages amount here',
                            key='stage_amount',
                            min_value=1)
            st.write('Numbers vectors\' size')
            vectors_size_input = st.number_input(
                label='Enter vectors\' size here',
                key='vectors_size',
                min_value=1
            )
            if vectors_size_input:
                columns = st.columns(2)
                with columns[0]:
                    st.write('Vector 1')
                with columns[1]:
                    st.write('Vector 2')
                for i in range(vectors_size_input):
                    with columns[0]:
                        st.number_input(label=f'Vector 1 Number {i+1}',
                                        key=f'vector_1_number_{i}',
                                        min_value=0)
                    with columns[1]:
                        st.number_input(label=f'Vector 2 Number {i+1}',
                                        key=f'vector_2_number_{i}',
                                        min_value=0)
            create_pipeline_btn = st.button(label='Create arithmetic pipeline',
                                            key='create_pipeline',
                                            on_click=self._create_pipeline)

    def _render_pipeline_tacts(self):
        with open(os.path.realpath(os.path.join(
            os.path.dirname(__file__), 'src/pipeline_work.json'
        ))) as file:
            pipeline_work_story = json.load(file)
        if st.session_state.get('rendered_tact') is None:
            st.session_state.rendered_tact = 0                   
        self._render_pipline_tact(pipeline_work_story,
                                  st.session_state.get('rendered_tact', 0))

        
    
    def _render_pipline_tact(self, work_story: list[dict], tact_number: int =
                             st.session_state.get('rendered_tact', 0)):
        tact = work_story[tact_number]        
        with self._placeholder.container(border=False):
            st.write(f'Tact #{tact['tacts_done']}')
            st.write(f'Queue 1: {tact['queue_1']}')
            st.write(f'Queue 2: {tact['queue_2']}')
            for index, level in tact['levels_status'].items():
                st.write(f'Stage {int(index) + 1}')
                st.write(f'Pair index: {'--' if (pair_index := level['pair_index'])
                                        is None else pair_index}')
                st.write(f'Partial sum: {level['status']['partial_sum']} | '
                         f'Partial product: {level['status']['partial_product']}')
            st.write(f'Result vector: {tact['result']}')
            if tact_number == 0:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.button(label='Next tact ▶️', key='move_next',
                              on_click=lambda: self._change_tact(next=True),
                              use_container_width=True)
                with col2:
                    st.button('Create new pipeline', key='new_pipeline',
                              on_click=self._new_pipeline,
                              use_container_width=True)
            elif 0 < tact_number < len(work_story) - 1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button(label='Previous tact ◀️', key='move_previous',
                              on_click=lambda: self._change_tact(next=False),
                              use_container_width=True)
                with col2:
                    st.button(label='Next tact ▶️', key='move_next',
                              on_click=lambda: self._change_tact(next=True),
                              use_container_width=True)
                with col3:
                    st.button('Create new pipeline', key='new_pipeline',
                              on_click=self._new_pipeline,
                              use_container_width=True)
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.button(label='Previous tact ◀️', key='move_previous',
                              on_click=lambda: self._change_tact(next=False),
                              use_container_width=True)
                with col2:
                    st.button('Create new pipeline', key='new_pipeline',
                              on_click=self._new_pipeline,
                              use_container_width=True)

    def _create_pipeline(self):
        pipeline = ArithmeticPipelineToJSON(
            vector_1=[st.session_state.get(f'vector_1_number_{i}')
                      for i in range(st.session_state.get('vectors_size'))],
            vector_2=[st.session_state.get(f'vector_2_number_{i}')
                      for i in range(st.session_state.get('vectors_size'))],
            levels_amount=st.session_state.get('stage_amount'),
            number_bit_amount=6
        )
        pipeline.to_json()
        st.session_state.app_status = 'render_tacts'
    
    def _new_pipeline(self):
        st.session_state.rendered_tact = 0
        st.session_state.app_status = 'render_parameters'
    
    def _change_tact(self, next=True):
        if st.session_state.get('rendered_tact') is None:
            return
        if next:
            st.session_state.rendered_tact += 1
        else:
            st.session_state.rendered_tact -= 1


view = MainView()
view.run()
