// @vitest-environment node
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'

// Shape test only — we do not connect to Postgres here. The goal is
// to catch accidental edits that drop the new column or rename it.
const sql = readFileSync(
  path.resolve(__dirname, '../../lib/db/010_fallback_keypoints_json.sql'),
  'utf8',
)

describe('migration 010_fallback_keypoints_json.sql', () => {
  it('adds fallback_keypoints_json jsonb to user_sessions via ADD COLUMN IF NOT EXISTS', () => {
    expect(sql).toMatch(
      /ALTER TABLE user_sessions[\s\S]*ADD COLUMN IF NOT EXISTS\s+fallback_keypoints_json\s+jsonb/i,
    )
  })

  it('does not mark the new column NOT NULL (existing rows must remain valid)', () => {
    // The column must be nullable so existing user_sessions rows (which
    // have no fallback) don't violate the constraint when the migration
    // is applied to a populated database.
    expect(sql).not.toMatch(/fallback_keypoints_json[^\n]*NOT NULL/i)
  })

  it('does not alter the status enum (extracting was already allowed in schema.sql)', () => {
    // We deliberately rely on the existing CHECK constraint on
    // user_sessions.status, which already covers 'extracting'. If we
    // ever change that, this test forces a deliberate update.
    expect(sql).not.toMatch(/ALTER\s+TABLE\s+user_sessions[^\n]*status/i)
  })

  it('documents a down migration in a comment for manual rollback', () => {
    // We don't ship an automated runner; the rollback statement lives
    // in a comment so an operator can copy/paste it.
    expect(sql).toMatch(
      /DROP COLUMN IF EXISTS fallback_keypoints_json/i,
    )
  })
})
