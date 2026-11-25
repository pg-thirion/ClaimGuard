import pytest
from streamlit.testing.v1 import AppTest

@pytest.fixture(scope="module")
def app():
    return AppTest.from_file("app.py",default_timeout=90).run()

@pytest.mark.parametrize("page",["Portfolio overview","Investigation queue","Claim inspector","Case journal","Model evaluation","Threshold planner","Drift & scenarios","Model governance"])
def test_page(app,page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title[0].value==page

def test_case_flow(app):
    app.sidebar.radio[0].set_value("Claim inspector").run()
    app.text_area[0].set_value("Synthetic test note")
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    assert any("recorded" in x.value for x in app.success)
    app.sidebar.radio[0].set_value("Case journal").run()
    assert not app.exception
    assert any(x.value=="Verified" for x in app.metric)

def test_queue_controls(app):
    app.sidebar.radio[0].set_value("Investigation queue").run()
    app.selectbox[0].set_value("All claims")
    app.text_input[0].set_value("nonexistent")
    app.run()
    assert not app.exception
    assert any("0 matching claims" in x.value for x in app.markdown)
