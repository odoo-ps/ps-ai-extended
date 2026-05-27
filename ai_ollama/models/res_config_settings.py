from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    ollama_url = fields.Char(config_parameter="ai.ollama_url")
