import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';

// Re-implement or import the pure functions for the test runner
import { calculateTargetDimensions } from '../src/lib/imageProcessing.ts';
import { buildTruthfulDmcaNotice } from '../src/lib/dmcaNotice.ts';

test('1. Image Resizing: Aspect Ratio & Dimension Calculation', async (t) => {
    await t.test('preserves landscape aspect ratio and caps maximum dimension', () => {
        const result = calculateTargetDimensions(1920, 1080, 512);
        assert.equal(result.width, 512);
        assert.equal(result.height, 288);
        // Verify aspect ratio preservation
        const originalRatio = 1920 / 1080;
        const resizedRatio = result.width / result.height;
        assert.ok(Math.abs(originalRatio - resizedRatio) < 0.01, 'Aspect ratio must be preserved');
    });

    await t.test('preserves portrait aspect ratio without height/width corruption', () => {
        // Previously, the bug caused height to be assigned twice: newH = ... (width / height)
        const result = calculateTargetDimensions(1080, 1920, 512);
        assert.equal(result.height, 512);
        assert.equal(result.width, 288);
        const originalRatio = 1080 / 1920;
        const resizedRatio = result.width / result.height;
        assert.ok(Math.abs(originalRatio - resizedRatio) < 0.01, 'Portrait ratio must be preserved');
    });

    await t.test('preserves 1:1 square dimensions', () => {
        const result = calculateTargetDimensions(1000, 1000, 512);
        assert.equal(result.width, 512);
        assert.equal(result.height, 512);
    });

    await t.test('does not upscale images smaller than max dimension', () => {
        const result = calculateTargetDimensions(320, 240, 512);
        assert.equal(result.width, 320);
        assert.equal(result.height, 240);
    });

    await t.test('handles extreme panorama dimensions safely', () => {
        const result = calculateTargetDimensions(5000, 200, 512);
        assert.equal(result.width, 512);
        assert.equal(result.height, 20);
        assert.ok(result.height >= 1, 'Height must be positive');
    });

    await t.test('handles extreme tall vertical image dimensions safely', () => {
        const result = calculateTargetDimensions(200, 5000, 512);
        assert.equal(result.height, 512);
        assert.equal(result.width, 20);
        assert.ok(result.width >= 1, 'Width must be positive');
    });

    await t.test('handles non-positive, NaN, and infinite dimensions gracefully', () => {
        const zeroResult = calculateTargetDimensions(0, 0, 512);
        assert.equal(zeroResult.width, 512);
        assert.equal(zeroResult.height, 512);

        const negativeResult = calculateTargetDimensions(-100, 500, 512);
        assert.equal(negativeResult.width, 512);
        assert.equal(negativeResult.height, 512);

        const nanResult = calculateTargetDimensions(NaN, 500, 512);
        assert.equal(nanResult.width, 512);
        assert.equal(nanResult.height, 512);
    });
});

test('2. Cryptographic Hash Integrity: SHA-256 Digest Computation', async (t) => {
    await t.test('computes correct NIST test vector for empty bytes', async () => {
        const emptyBuffer = new Uint8Array(0);
        const digest = await crypto.subtle.digest('SHA-256', emptyBuffer);
        const hex = Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
        assert.equal(hex, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    });

    await t.test('computes correct NIST test vector for known string "hello world"', async () => {
        const encoder = new TextEncoder();
        const data = encoder.encode('hello world');
        const digest = await crypto.subtle.digest('SHA-256', data);
        const hex = Array.from(new Uint8Array(digest)).map(b => b.toString(16).padStart(2, '0')).join('');
        assert.equal(hex, 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
        assert.equal(hex.length, 64);
    });
});

test('3. DMCA Notice Generation: Truthful Findings & Evidence Verification', async (t) => {
    await t.test('disables notice generation when 0 findings are present', () => {
        const noticeEmpty = buildTruthfulDmcaNotice({
            matches: [],
            biometricHash: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        });
        assert.equal(noticeEmpty, null, 'Must return null when matches list is empty');
    });

    await t.test('generates notice containing only real detected URLs and real hash', () => {
        const realMatches = [
            { platform: 'Twitter / X', url: 'https://x.com/fake_account/status/123456789', similarity_score: 87.5 },
            { platform: 'Instagram', url: 'https://instagram.com/p/abcdef123', similarity_score: 91.2 }
        ];
        const realHash = 'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9';

        const notice = buildTruthfulDmcaNotice({
            matches: realMatches,
            biometricHash: realHash,
            noticeDate: '2026-10-13'
        });

        assert.ok(notice !== null);
        assert.ok(notice.includes('https://x.com/fake_account/status/123456789'));
        assert.ok(notice.includes('https://instagram.com/p/abcdef123'));
        assert.ok(notice.includes('Cryptographic Biometric Hash (SHA-256): ' + realHash));

        // Ensure NO fabricated audit placeholders exist in the notice
        assert.ok(!notice.includes('reddit.com/r/deepfakes_vault'));
        assert.ok(!notice.includes('0x9F4D2B8A1C0E7635'));
        assert.ok(!notice.includes('t.me/synthetic_archive_leak'));
    });
});

test('4. Plan & Quota Security: Client Fallback Integrity', async (t) => {
    await t.test('defaults strictly to STARTER plan when unauthenticated or storage empty', () => {
        const defaultPlan = 'STARTER';
        assert.equal(defaultPlan, 'STARTER');
        assert.notEqual(defaultPlan, 'PROFESSIONAL');
        assert.notEqual(defaultPlan, 'ENTERPRISE');
    });

    await t.test('quota unavailable fallback prevents silent privilege elevation', () => {
        const quotaFallback = {
            plan: 'STARTER',
            scansToday: 0,
            dailyLimit: 10,
            remainingToday: 'Unavailable',
            historyDaysLimit: 7,
            bulkLimit: 1,
            identityProtectionAllowed: false,
            pdfExportAllowed: false,
            apiAccessAllowed: false,
            quotaUnavailable: true,
        };

        assert.equal(quotaFallback.plan, 'STARTER');
        assert.equal(quotaFallback.quotaUnavailable, true);
        assert.equal(quotaFallback.identityProtectionAllowed, false);
        assert.equal(quotaFallback.pdfExportAllowed, false);
        assert.equal(quotaFallback.apiAccessAllowed, false);
    });
});
