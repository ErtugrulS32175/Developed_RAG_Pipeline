import logging
logging.getLogger('docling').setLevel(logging.DEBUG)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

engine_options = ApiVlmEngineOptions(
    runtime_type=VlmEngineType.API,
    url='http://localhost:8003/v1/chat/completions',
    params=dict(
        model='ibm-granite/granite-docling-258M',
        max_tokens=4096,
        temperature=0.0,
        skip_special_tokens=False,
    ),
    timeout=90,
    response_format='doctags',
)
vlm_options = VlmConvertOptions.from_preset('granite_docling', engine_options=engine_options)
pipeline_options = VlmPipelineOptions(generate_page_images=True, vlm_options=vlm_options, enable_remote_services=True)
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=pipeline_options)})

result = converter.convert('./data/page13.pdf', raises_on_error=False, max_num_pages=1)

for page in result.pages:
    if hasattr(page, 'predictions') and page.predictions:
        if hasattr(page.predictions, 'vlm_response') and page.predictions.vlm_response:
            print(f'Page {page.page_no}:')
            print(page.predictions.vlm_response.text[:300])
            break

md = result.document.export_to_markdown()
print('Length:', len(md))
