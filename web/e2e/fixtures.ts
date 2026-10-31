import { expect, test as base } from "@playwright/test";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

/**
 * すべての試験の前に、初期設定ウィザードを終えた状態にする(初回起動のウィザードが他の画面を隠すため)。
 * ウィザード自体の試験(wizard.spec.ts)は、設定を空に戻してから始める。
 * 設定を PUT する試験は、`wizard_done: true` を含めること(含めないとウィザードがまた出る)。
 */
export const test = base.extend<{ wizardDone: void }>({
  wizardDone: [
    async ({ request }, use) => {
      const res = await request.get("/api/settings", { headers: AUTH });
      expect(res.ok()).toBeTruthy();
      const { settings } = (await res.json()) as { settings: Record<string, unknown> };
      if (!settings.wizard_done) {
        const put = await request.put("/api/settings", { headers: AUTH, data: { ...settings, wizard_done: true } });
        expect(put.ok()).toBeTruthy();
      }
      await use();
    },
    { auto: true },
  ],
});

export { expect };
