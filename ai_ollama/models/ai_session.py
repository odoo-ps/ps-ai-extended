from odoo import api, fields, models

from odoo.addons.ai.services.ai_provider import AIProvider


class AISession(models.TransientModel):
    _inherit = "ai.session"

    llm_model = fields.Selection(
        selection="_get_llm_model_selection",
        string="LLM Model",
    )

    @api.model
    def _get_llm_model_selection(self):
        return AIProvider.get_all_llm_models()

    def _generate_channel_name(self, message):
        self.ensure_one()
        if AIProvider.get_by_model(self.env, self.llm_model).name != 'ollama':
            return super()._generate_channel_name(message)

        if not (channel := self.channel_id):
            raise Exception(self.env._("The session is not linked to a discuss channel"))

        channel.sudo().write({'name': "Title"})
