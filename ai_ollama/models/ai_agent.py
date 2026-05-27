from odoo import api, fields, models

from odoo.addons.ai.services.ai_provider import AIProvider


class AIAgent(models.Model):
    _inherit = "ai.agent"

    llm_model = fields.Selection(
        selection="_get_llm_model_selection",
        string="LLM Model",
        default="gpt-4o",
        required=True,
    )

    @api.model
    def _get_llm_model_selection(self):
        return AIProvider.get_all_llm_models()
