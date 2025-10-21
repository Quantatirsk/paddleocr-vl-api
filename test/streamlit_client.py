"""
极简 Streamlit OCR 客户端
"""
import streamlit as st
import requests
import zipfile
import io
import json

API_BASE = "http://localhost:8781"

st.set_page_config(page_title="PaddleOCR", page_icon="📄", layout="wide")

# 添加自定义 CSS 控制内容宽度和间距
st.markdown("""
<style>
    .stMarkdown {max-width: 100%; overflow-x: auto;}
    .stMarkdown img {max-width: 100%; height: auto;}
    .stMarkdown table {max-width: 100%; overflow-x: auto; display: block;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {margin-bottom: 1rem !important;}
    h2 {margin-top: 1rem !important; margin-bottom: 0.8rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("📄 PaddleOCR")

uploaded_files = st.file_uploader(
    "上传文件 (可多选)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

# 选项配置
col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1.5, 1, 1])
with col1:
    return_md = st.checkbox("📝 MD", value=True)
with col2:
    return_json = st.checkbox("📋 JSON", value=False)
with col3:
    return_images = st.checkbox("🖼️ 图片", value=False)
with col4:
    response_zip = st.checkbox("📦 ZIP", value=False)
with col5:
    start_page = st.number_input("起始页", min_value=0, value=0)
with col6:
    end_page = st.number_input("结束页", min_value=0, value=99999)

submit = st.button("🚀 开始识别", type="primary")

if submit:
    if not uploaded_files:
        st.warning("⚠️ 请先上传文件")
    else:
        try:
            with st.spinner("⏳ 处理中..."):
                # 构建文件列表
                files = []
                if isinstance(uploaded_files, list):
                    for f in uploaded_files:
                        files.append(('files', (f.name, f.getvalue())))
                else:
                    files = [('files', (uploaded_files.name, uploaded_files.getvalue()))]

                data = {
                    'return_md': return_md,
                    'return_middle_json': return_json,
                    'return_images': return_images,
                    'response_format_zip': response_zip,
                    'start_page_id': start_page,
                    'end_page_id': end_page
                }

                r = requests.post(f"{API_BASE}/file_parse", files=files, data=data)

                if r.ok:
                    st.success("✅ 处理成功")

                    if response_zip:
                        # ZIP 下载
                        st.download_button(
                            "📥 下载 ZIP",
                            data=r.content,
                            file_name="ocr_result.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                        # 显示 ZIP 内容
                        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                            st.info(f"📦 ZIP 包含 {len(zf.namelist())} 个文件")
                            with st.expander("查看文件列表"):
                                for name in sorted(zf.namelist()):
                                    info = zf.getinfo(name)
                                    st.text(f"{'📁' if name.endswith('/') else '📄'} {name} ({info.file_size:,} bytes)")
                    else:
                        # JSON 响应
                        result = r.json()

                        # 显示每个文件的结果
                        for fname, fdata in result.get('results', {}).items():
                            st.subheader(f"📄 {fname}")

                            if fdata.get('md_content'):
                                with st.expander("📝 Markdown 内容", expanded=True):
                                    st.markdown(fdata['md_content'], unsafe_allow_html=True)

                            if fdata.get('middle_json'):
                                with st.expander("📋 JSON 数据"):
                                    st.json(json.loads(fdata['middle_json']))

                            if fdata.get('images'):
                                with st.expander(f"🖼️ 图片 ({len(fdata['images'])} 张)"):
                                    cols = st.columns(3)
                                    for idx, (img_name, img_data) in enumerate(fdata['images'].items()):
                                        with cols[idx % 3]:
                                            st.image(img_data, caption=img_name, use_container_width=True)

                            st.divider()
                else:
                    st.error(f"❌ 错误: {r.status_code}")
                    st.code(r.text)

        except Exception as e:
            st.error(f"❌ 异常: {e}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
