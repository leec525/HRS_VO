from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, FloatField, BooleanField, RadioField, SelectField, BooleanField
from wtforms.validators import DataRequired, Length, Email

class MtmctForm(FlaskForm):
    bSave = BooleanField('保存视频')
    setting_path = StringField(u'配置文件路径')
    submit = SubmitField(u'开始')

class DispForm(FlaskForm):
    submit = SubmitField('暂停/继续')