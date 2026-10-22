const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

export interface Conversation {
  id: string;
  title: string;
  status: string;
  document_count: number;
  message_count: number;
  created_at: string;
}

export interface Document {
  id: string;
  filename: string;
  status: string;
  chunk_count: number;
  error_code?: string;
  error_message?: string;
}

export interface Citation {
  evidence_id: string;
  filename: string;
  title: string;
  page_start: number | null;
  page_end: number | null;
  breadcrumb: string | null;
  quote: string;
  quote_truncated: boolean;
}

export interface Message {
  message_id: string;
  question?: string;
  answer_text?: string;
  status: string;
  citations: Citation[];
  role?: "user" | "assistant";
  content?: string;
}

export async function getConversations(): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/conversations`);
  if (!res.ok) throw new Error("Failed to fetch conversations");
  const data = await res.json();
  return data.conversations;
}

export async function createConversation(title: string): Promise<Conversation> {
  const res = await fetch(`${API_BASE}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to create conversation");
  return res.json();
}

export async function getDocuments(convId: string): Promise<Document[]> {
  const res = await fetch(`${API_BASE}/conversations/${convId}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  const data = await res.json();
  return data.documents;
}

// --- UPDATED FOR MULTIPLE FILES ---
export async function uploadDocuments(convId: string, files: File[]): Promise<void> {
  const formData = new FormData();
  // Append all files with the same key "files"
  files.forEach((file) => {
    formData.append("files", file);
  });
  
  const res = await fetch(`${API_BASE}/conversations/${convId}/documents`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message || "Upload failed");
  }
}

export async function askQuestion(convId: string, question: string): Promise<Message> {
  const res = await fetch(`${API_BASE}/conversations/${convId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message || "Failed to get answer");
  }
  return res.json();
}

export async function getMessages(convId: string): Promise<any[]> {
  const res = await fetch(`${API_BASE}/conversations/${convId}/messages`);
  if (!res.ok) throw new Error("Failed to fetch messages");
  const data = await res.json();
  return data.messages;
}

export async function deleteConversation(convId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${convId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message || "Failed to delete conversation");
  }
}

export async function deleteDocument(convId: string, docId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/conversations/${convId}/documents/${docId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error?.message || "Failed to delete document");
  }
}
export async function renameConversation(convId: string, title: string): Promise<void> {
    const res = await fetch(`${API_BASE}/conversations/${convId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error?.message || "Failed to rename conversation");
    }
  }