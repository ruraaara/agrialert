#!/usr/bin/env python3
"""
TaniBot Offline Chatbot Engine
===============================
Rule-based decision-tree chatbot untuk HarvestAlert.
Membaca database SQLite dan mengembalikan respons terstruktur.

Fitur:
- Post-detection recommendation flow (triggered setelah CNN pest detection)
- Multiple-choice navigation
- FAQ search
- Sumber tercantum di setiap respons

Usage:
    engine = TaniBotEngine("tanibot_offline.db")
    response = engine.start()
    response = engine.select("opt_detection")
    response = engine.get_post_detection_recs("blas", "pertanaman")
"""

import sqlite3
import json
import os


class TaniBotResponse:
    """Structured response dari chatbot."""
    def __init__(self, text, options=None, recs=None, sumber=None, node_id=None):
        self.text = text
        self.options = options or []      # list of (node_id, label)
        self.recs = recs or []            # list of dict for recommendations
        self.sumber = sumber or []        # list of source strings
        self.node_id = node_id

    def to_dict(self):
        return {
            "text": self.text,
            "options": [{"id": o[0], "label": o[1]} for o in self.options],
            "recommendations": self.recs,
            "sumber": self.sumber,
            "node_id": self.node_id,
        }

    def __repr__(self):
        parts = [f"\n{'='*60}", self.text, f"{'='*60}"]
        if self.options:
            parts.append("\nPilihan:")
            for i, (nid, label) in enumerate(self.options, 1):
                parts.append(f"  [{i}] {label}")
        if self.recs:
            parts.append("\n📋 Rekomendasi Penanganan:")
            for j, r in enumerate(self.recs, 1):
                parts.append(f"\n  {j}. [{r.get('metode','')}] {r.get('langkah','')}")
                parts.append(f"     {r.get('detail','')}")
                if r.get("dosis"):
                    parts.append(f"     💊 Dosis: {r['dosis']}")
                if r.get("waktu_aplikasi"):
                    parts.append(f"     🕐 Waktu: {r['waktu_aplikasi']}")
        if self.sumber:
            parts.append(f"\n📚 Sumber: {'; '.join(set(self.sumber))}")
        return "\n".join(parts)


class TaniBotEngine:
    """Offline chatbot engine backed by SQLite."""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "tanibot_offline.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.current_node = "root"

    def close(self):
        self.conn.close()

    # ----------------------------------------------------------------
    # Core navigation
    # ----------------------------------------------------------------
    def start(self):
        """Return the root menu."""
        self.current_node = "root"
        return self._get_node_response("root")

    def select(self, node_id):
        """User selects an option -> navigate to that node."""
        c = self.conn.cursor()
        node = c.execute("SELECT * FROM decision_tree WHERE node_id=?", (node_id,)).fetchone()
        if not node:
            return TaniBotResponse("❌ Pilihan tidak ditemukan. Silakan coba lagi.", 
                                   options=[("root", "🏠 Kembali ke menu utama")])

        self.current_node = node_id
        tipe = node["tipe"]

        if tipe == "action":
            return self._handle_action(node)
        else:
            return self._get_node_response(node_id)

    def select_by_index(self, index):
        """Select by 1-based index from current options."""
        c = self.conn.cursor()
        children = c.execute(
            "SELECT node_id, teks FROM decision_tree WHERE parent_id=? ORDER BY rowid",
            (self.current_node,)
        ).fetchall()
        if 1 <= index <= len(children):
            return self.select(children[index - 1]["node_id"])
        return TaniBotResponse(f"❌ Pilihan {index} tidak valid. Pilih 1-{len(children)}.",
                               options=[(ch["node_id"], ch["teks"]) for ch in children],
                               node_id=self.current_node)

    def _get_node_response(self, node_id):
        """Build response for a given node (show text + children as options)."""
        c = self.conn.cursor()
        node = c.execute("SELECT * FROM decision_tree WHERE node_id=?", (node_id,)).fetchone()
        if not node:
            return TaniBotResponse("Node not found.")

        children = c.execute(
            "SELECT node_id, teks FROM decision_tree WHERE parent_id=? ORDER BY rowid",
            (node_id,)
        ).fetchall()

        options = [(ch["node_id"], ch["teks"]) for ch in children]
        return TaniBotResponse(node["teks"], options=options, node_id=node_id)

    def _handle_action(self, node):
        """Handle action nodes that query the database."""
        metadata = json.loads(node["metadata"]) if node["metadata"] else {}
        opt_id = node["opt_id"]

        # === Rekomendasi pengendalian per fase ===
        if opt_id and "fase" in metadata:
            return self.get_post_detection_recs(opt_id, metadata["fase"])

        # === FAQ by kategori ===
        if "kategori" in metadata:
            return self._get_faq_by_kategori(metadata["kategori"])

        # === Info OPT (non-CNN: PBP, WBC, Tikus) ===
        if metadata.get("tipe") == "info_opt" and opt_id:
            return self._get_opt_info_card(opt_id)

        # === Pemupukan ===
        if metadata.get("tipe") == "pemupukan":
            return self._get_pemupukan()

        # === Varietas ===
        if metadata.get("tipe") == "varietas":
            return self._get_varietas(metadata.get("opt_id"))

        return TaniBotResponse("⚠️ Aksi tidak dikenali.", options=[("root", "🏠 Menu utama")])

    # ----------------------------------------------------------------
    # Post-detection recommendation (MAIN FEATURE)
    # ----------------------------------------------------------------
    def get_post_detection_recs(self, opt_id, fase):
        """
        Main post-detection flow: return structured recommendations 
        for a detected pest/disease at a given growth phase.
        Called directly after CNN detection or via decision tree.
        """
        c = self.conn.cursor()

        # Get OPT info
        opt = c.execute("SELECT * FROM opt WHERE id=?", (opt_id,)).fetchone()
        if not opt:
            return TaniBotResponse(f"❌ OPT '{opt_id}' tidak ditemukan di database.")

        # Get recommendations for this OPT + phase
        recs = c.execute("""
            SELECT * FROM rekomendasi 
            WHERE opt_id=? AND fase=? 
            ORDER BY prioritas ASC
        """, (opt_id, fase)).fetchall()

        # Get varietas tahan if available
        varietas = c.execute(
            "SELECT nama_varietas, ketahanan FROM varietas_tahan WHERE opt_id=?", 
            (opt_id,)
        ).fetchall()

        # Build response text
        fase_labels = {
            "pratanam": "Pratanam / Persiapan Lahan",
            "pesemaian": "Pesemaian",
            "pertanaman": "Pertanaman"
        }
        text_parts = [
            f"🚨 Terdeteksi: {opt['nama_lokal']}",
            f"📛 ({opt['nama_latin']})",
            f"📊 Tipe: {'Hama' if opt['tipe']=='hama' else 'Penyakit'}",
            f"\n📝 Gejala:\n{opt['gejala']}",
            f"\n⚠️ Ambang pengendalian:\n{opt['ambang_pengendalian']}",
            f"\n📌 Fase saat ini: {fase_labels.get(fase, fase)}",
            f"\n{'─'*40}",
            f"📋 REKOMENDASI PENANGANAN ({len(recs)} langkah):"
        ]

        rec_dicts = []
        sumber_set = set()
        for r in recs:
            rd = {
                "metode": r["metode"].replace("_", " ").title(),
                "langkah": r["langkah"],
                "detail": r["detail"],
                "dosis": r["dosis"],
                "waktu_aplikasi": r["waktu_aplikasi"],
                "prioritas": r["prioritas"],
            }
            rec_dicts.append(rd)
            sumber_set.add(r["sumber"])

        # Add varietas info
        if varietas:
            text_parts.append(f"\n🌱 Varietas tahan {opt['nama_lokal']}:")
            for v in varietas:
                text_parts.append(f"  • {v['nama_varietas']} — {v['ketahanan']}")

        text = "\n".join(text_parts)

        # Add nav options
        options = [
            ("root", "🏠 Kembali ke menu utama"),
        ]
        # Add FAQ link for related OPT
        kategori_map = {
            "blas": "faq_blas", "wbc": "faq_wereng",
            "tikus": "faq_tikus", "hdb": "faq_umum",
            "tungro": "faq_umum", "pbp": "faq_umum"
        }
        if opt_id in kategori_map:
            options.insert(0, (kategori_map[opt_id], f"❓ FAQ terkait {opt['nama_lokal']}"))

        return TaniBotResponse(
            text=text, options=options,
            recs=rec_dicts, sumber=list(sumber_set),
            node_id=f"ans_{opt_id}_{fase}"
        )

    # ----------------------------------------------------------------
    # Info card for non-CNN OPT (PBP, WBC, Tikus)
    # ----------------------------------------------------------------
    def _get_opt_info_card(self, opt_id):
        """Show full OPT info + all recs across all phases. For non-CNN OPT."""
        c = self.conn.cursor()
        opt = c.execute("SELECT * FROM opt WHERE id=?", (opt_id,)).fetchone()
        if not opt:
            return TaniBotResponse(f"OPT '{opt_id}' tidak ditemukan.",
                                   options=[("root", "🏠 Menu utama")])

        text_parts = [
            f"📖 Info: {opt['nama_lokal']}",
            f"({opt['nama_latin']})",
            f"Tipe: {'Hama' if opt['tipe']=='hama' else 'Penyakit'}",
            f"\n📝 Deskripsi:\n{opt['deskripsi']}",
            f"\n🔍 Gejala:\n{opt['gejala']}",
            f"\n⚠️ Ambang pengendalian:\n{opt['ambang_pengendalian']}",
            f"\n⚠️ Catatan: Hama ini tidak terdeteksi melalui foto daun (CNN).",
            f"Jika Anda menduga serangan {opt['nama_lokal']}, laporkan ke petugas POPT/PPL setempat.",
        ]

        # Get all recs grouped by phase
        recs = c.execute(
            "SELECT * FROM rekomendasi WHERE opt_id=? ORDER BY fase, prioritas", (opt_id,)
        ).fetchall()

        rec_dicts = []
        sumber_set = set()
        if recs:
            text_parts.append(f"\n{'─'*40}")
            text_parts.append(f"📋 REKOMENDASI PENGENDALIAN:")
            current_fase = None
            for r in recs:
                if r["fase"] != current_fase:
                    current_fase = r["fase"]
                    fase_label = {"pratanam":"Pratanam","pesemaian":"Pesemaian","pertanaman":"Pertanaman"}.get(current_fase, current_fase)
                    text_parts.append(f"\n▸ Fase {fase_label}:")
                rd = {
                    "metode": r["metode"].replace("_", " ").title(),
                    "langkah": r["langkah"], "detail": r["detail"],
                    "dosis": r["dosis"], "waktu_aplikasi": r["waktu_aplikasi"],
                    "prioritas": r["prioritas"],
                }
                rec_dicts.append(rd)
                sumber_set.add(r["sumber"])

        # Varietas
        varietas = c.execute(
            "SELECT nama_varietas, ketahanan FROM varietas_tahan WHERE opt_id=?", (opt_id,)
        ).fetchall()
        if varietas:
            text_parts.append(f"\n🌱 Varietas tahan:")
            for v in varietas:
                text_parts.append(f"  • {v['nama_varietas']} — {v['ketahanan']}")

        sumber_set.add(opt["sumber"])

        return TaniBotResponse(
            text="\n".join(text_parts),
            options=[("root", "🏠 Menu utama"), ("opt_info_hama", "📖 Info hama lain")],
            recs=rec_dicts, sumber=list(sumber_set),
            node_id=f"info_{opt_id}"
        )

    # ----------------------------------------------------------------
    # FAQ
    # ----------------------------------------------------------------
    def _get_faq_by_kategori(self, kategori):
        c = self.conn.cursor()
        faqs = c.execute(
            "SELECT * FROM faq WHERE kategori=? ORDER BY id", (kategori,)
        ).fetchall()

        if not faqs:
            return TaniBotResponse(f"Tidak ada FAQ untuk kategori '{kategori}'.",
                                   options=[("root", "🏠 Menu utama")])

        text_parts = [f"❓ FAQ — {kategori.replace('_',' ').title()}\n{'─'*40}"]
        sumber_set = set()
        for i, f in enumerate(faqs, 1):
            text_parts.append(f"\n{i}. {f['pertanyaan']}")
            text_parts.append(f"   {f['jawaban']}")
            sumber_set.add(f["sumber"])

        return TaniBotResponse(
            text="\n".join(text_parts),
            options=[("root", "🏠 Kembali ke menu utama"), ("opt_faq", "❓ Topik FAQ lain")],
            sumber=list(sumber_set)
        )

    def search_faq(self, query):
        """Simple keyword search across all FAQ."""
        c = self.conn.cursor()
        results = c.execute("""
            SELECT * FROM faq 
            WHERE pertanyaan LIKE ? OR jawaban LIKE ? OR kategori LIKE ?
            ORDER BY id LIMIT 5
        """, (f"%{query}%", f"%{query}%", f"%{query}%")).fetchall()

        if not results:
            return TaniBotResponse(
                f"🔍 Tidak ditemukan FAQ untuk '{query}'. Coba kata kunci lain.",
                options=[("root", "🏠 Menu utama")]
            )

        text_parts = [f"🔍 Hasil pencarian: '{query}'\n{'─'*40}"]
        sumber_set = set()
        for i, f in enumerate(results, 1):
            text_parts.append(f"\n{i}. {f['pertanyaan']}")
            text_parts.append(f"   {f['jawaban']}")
            sumber_set.add(f["sumber"])

        return TaniBotResponse(
            text="\n".join(text_parts),
            options=[("root", "🏠 Menu utama")],
            sumber=list(sumber_set)
        )

    # ----------------------------------------------------------------
    # Pemupukan
    # ----------------------------------------------------------------
    def _get_pemupukan(self):
        c = self.conn.cursor()
        rows = c.execute("SELECT * FROM pemupukan ORDER BY id").fetchall()

        text_parts = [
            "🧪 REKOMENDASI PEMUPUKAN PADI SAWAH",
            "(Berdasarkan hasil uji tanah - Balai Penelitian Tanah)\n",
            "─" * 40
        ]
        sumber_set = set()
        for r in rows:
            text_parts.append(f"\n📦 {r['jenis_pupuk']}")
            text_parts.append(f"   Dosis: {r['dosis_per_ha']}")
            text_parts.append(f"   Waktu: {r['waktu_aplikasi']}")
            if r["catatan"]:
                text_parts.append(f"   📌 {r['catatan']}")
            sumber_set.add(r["sumber"])

        text_parts.append(f"\n{'─'*40}")
        text_parts.append("⚠️ Dosis di atas adalah rekomendasi umum.")
        text_parts.append("Lakukan UJI TANAH di lahan Anda untuk dosis yang tepat.")

        return TaniBotResponse(
            text="\n".join(text_parts),
            options=[("root", "🏠 Menu utama")],
            sumber=list(sumber_set)
        )

    # ----------------------------------------------------------------
    # Varietas
    # ----------------------------------------------------------------
    def _get_varietas(self, opt_id):
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT v.*, o.nama_lokal FROM varietas_tahan v JOIN opt o ON v.opt_id=o.id WHERE v.opt_id=? ORDER BY v.id",
            (opt_id,)
        ).fetchall()

        if not rows:
            return TaniBotResponse(f"Belum ada data varietas tahan untuk OPT ini.",
                                   options=[("root", "🏠 Menu utama")])

        opt_nama = rows[0]["nama_lokal"]
        text_parts = [f"🌱 VARIETAS TAHAN {opt_nama.upper()}\n{'─'*40}"]
        sumber_set = set()
        for r in rows:
            text_parts.append(f"  • {r['nama_varietas']} — {r['ketahanan']}")
            sumber_set.add(r["sumber"])

        return TaniBotResponse(
            text="\n".join(text_parts),
            options=[("root", "🏠 Menu utama"), ("opt_varietas", "🌱 Varietas OPT lain")],
            sumber=list(sumber_set)
        )

    # ----------------------------------------------------------------
    # Direct API for app integration
    # ----------------------------------------------------------------
    def get_opt_info(self, opt_id):
        """Get full OPT info by ID. For app integration."""
        c = self.conn.cursor()
        opt = c.execute("SELECT * FROM opt WHERE id=?", (opt_id,)).fetchone()
        if opt:
            return dict(opt)
        return None

    def get_all_opt(self):
        """List all OPT entries."""
        c = self.conn.cursor()
        return [dict(r) for r in c.execute("SELECT id, nama_lokal, tipe FROM opt ORDER BY id").fetchall()]

    def get_agens_hayati(self, target_opt=None):
        """Get biological control agents, optionally filtered by target."""
        c = self.conn.cursor()
        if target_opt:
            rows = c.execute(
                "SELECT * FROM agens_hayati WHERE target_opt LIKE ?", (f"%{target_opt}%",)
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM agens_hayati ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def get_db_stats(self):
        """Return database statistics for verification."""
        c = self.conn.cursor()
        stats = {}
        for table in ["opt", "rekomendasi", "varietas_tahan", "faq", "decision_tree", "agens_hayati", "pemupukan"]:
            stats[table] = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats["db_size_kb"] = os.path.getsize(self.db_path) / 1024
        return stats


# ================================================================
# CLI Demo
# ================================================================
def demo_cli():
    """Interactive CLI demo of TaniBot."""
    # Build DB if needed
    db_path = os.path.join(os.path.dirname(__file__), "tanibot_offline.db")
    if not os.path.exists(db_path):
        from tanibot_db_builder import build_database
        build_database()

    engine = TaniBotEngine(db_path)

    print("\n" + "=" * 60)
    print("  🌾 TaniBot Offline — Demo CLI")
    print("  Ketik nomor pilihan, 'cari <kata>', atau 'q' untuk keluar")
    print("=" * 60)

    # Show DB stats
    stats = engine.get_db_stats()
    print(f"\n📊 Database: {stats['db_size_kb']:.1f} KB")
    print(f"   OPT: {stats['opt']} | Rekomendasi: {stats['rekomendasi']} | FAQ: {stats['faq']}")
    print(f"   Varietas: {stats['varietas_tahan']} | Agens Hayati: {stats['agens_hayati']}")
    print(f"   Decision Tree Nodes: {stats['decision_tree']}")

    response = engine.start()
    print(response)

    while True:
        user = input("\n> ").strip()
        if user.lower() in ("q", "quit", "exit"):
            print("👋 Terima kasih! Selamat bertani.")
            break
        elif user.lower().startswith("cari "):
            query = user[5:].strip()
            response = engine.search_faq(query)
            print(response)
        elif user.isdigit():
            response = engine.select_by_index(int(user))
            print(response)
        elif user in ("0", "menu", "home"):
            response = engine.start()
            print(response)
        else:
            # Try as node_id
            response = engine.select(user)
            print(response)

    engine.close()


if __name__ == "__main__":
    demo_cli()
