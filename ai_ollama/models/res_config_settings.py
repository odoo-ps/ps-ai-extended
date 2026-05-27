from odoo import fields, models


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    ollama_url = fields.Char(
        string="Ollama Base URL",
        config_parameter="ai.ollama_url",
    )
