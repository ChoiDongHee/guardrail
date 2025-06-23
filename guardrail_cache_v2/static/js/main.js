/**
 * Main JavaScript for the Redis Vector Similarity Search app
 * 업데이트된 API 구조 지원
 */

// DOM Elements
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const chatMessages = document.getElementById('chat-messages');
const loadingIndicator = document.getElementById('loading-indicator');
const alertContainer = document.getElementById('alert-container');

// API Endpoints
const API_ENDPOINTS = {
    QUERY: '/api/query',
    DATA: '/api/data',
    STATS: '/api/stats'
};

// 검색 방법 설정
let searchMethod = 'vector'; // 'vector' 또는 'keyword'

/**
 * Display an alert message
 */
function showAlert(message, type = 'warning', timeout = 5000) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <span>${message}</span>
        <button type="button" class="alert-close" onclick="this.parentElement.remove()">&times;</button>
    `;

    alertContainer.appendChild(alert);

    if (timeout > 0) {
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, timeout);
    }
}

/**
 * Add a message to the chat interface
 */
function addMessage(content, sender, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${sender}`;

    // Message info
    const messageInfo = document.createElement('div');
    messageInfo.className = 'message-info';

    let infoText = sender === 'user' ? 'You' : 'Bot';

    // 캐시된 응답인 경우 표시
    if (sender === 'bot' && metadata.cached) {
        infoText += ' (cached)';
        messageDiv.classList.add('cached-response');

        // 유사도와 히트 수 표시
        if (metadata.similarity) {
            infoText += ` - 유사도: ${(metadata.similarity * 100).toFixed(1)}%`;
        }
        if (metadata.hits) {
            infoText += ` - 조회: ${metadata.hits}회`;
        }
    }

    messageInfo.textContent = infoText;

    // Message content
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    // 메시지 내용 처리 (개행 문자 변환)
    const formattedContent = content.replace(/\n/g, '<br>');
    messageContent.innerHTML = formattedContent;

    // Assemble message
    messageDiv.appendChild(messageInfo);
    messageDiv.appendChild(messageContent);

    // 검색 방법 표시 (봇 메시지인 경우)
    if (sender === 'bot' && metadata.method) {
        const methodBadge = document.createElement('span');
        methodBadge.className = 'search-method-badge';
        methodBadge.textContent = metadata.method === 'vector' ? '벡터검색' : '키워드검색';
        messageInfo.appendChild(methodBadge);
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Send a question to the API
 */
async function sendQuestion(question) {
    try {
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }

        // Add user message
        addMessage(question, 'user');

        // Prepare request
        const requestBody = {
            query: question,
            method: searchMethod
        };

        console.log('Sending request:', requestBody);

        // Send to API
        const response = await fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Received response:', data);

        // Handle response
        if (data.state) {
            if (data.cached && data.fresh_data) {
                // 캐시된 응답만 표시 (JSON 정보 숨김)
                addMessage(data.fresh_data.response, 'bot', {
                    cached: true,
                    similarity: data.similarity,
                    hits: data.fresh_data.hits,
                    method: searchMethod
                });

                // 유사한 질문도 표시 (질문이 다른 경우에만)
                if (data.fresh_data.question !== question) {
                    showAlert(
                        `💡 유사한 질문을 찾았습니다: "${data.fresh_data.question}"`,
                        'info',
                        6000
                    );
                }

                // 추가 정보 (선택적으로 표시)
                if (data.similarity === 1.0) {
                    const metaInfo = `📊 완벽한 매칭 | 📈 ${data.fresh_data.hits}회 조회됨`;
                    showAlert(metaInfo, 'success', 4000);
                }
            } else {
                // 캐시된 응답이 없는 경우
                let noResponseMessage = "죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다.";

                if (data.similarity > 0) {
                    noResponseMessage += ` (최고 유사도: ${(data.similarity * 100).toFixed(1)}%)`;
                }

                addMessage(noResponseMessage, 'bot', {
                    cached: false,
                    method: searchMethod
                });

                // 대안 제안
                showAlert(
                    `${searchMethod === 'vector' ? '키워드' : '벡터'} 검색으로 다시 시도해보시거나, 질문을 다르게 표현해보세요.`,
                    'info',
                    6000
                );
            }
        } else {
            // API 오류
            showAlert(`오류가 발생했습니다: ${data.error || '알 수 없는 오류'}`, 'danger');
            addMessage("시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", 'bot');
        }

    } catch (error) {
        console.error('Error sending question:', error);
        showAlert(`네트워크 오류: ${error.message}`, 'danger');
        addMessage("네트워크 오류가 발생했습니다. 연결을 확인해주세요.", 'bot');
    } finally {
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }
}

/**
 * Store a question-response pair
 */
async function storeQuestionResponse(question, method, response) {
    try {
        const result = await fetch(API_ENDPOINTS.DATA, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question,
                method: method || 'vector',
                answer: response
            })
        });

        if (!result.ok) {
            throw new Error(`HTTP ${result.status}: ${result.statusText}`);
        }

        const data = await result.json();

        if (data.state) {
            showAlert('데이터가 성공적으로 저장되었습니다!', 'success');
            return data.id;
        } else {
            showAlert(`저장 실패: ${data.error || '알 수 없는 오류'}`, 'danger');
            return null;
        }
    } catch (error) {
        console.error('Error storing data:', error);
        showAlert(`저장 오류: ${error.message}`, 'danger');
        return null;
    }
}

/**
 * Toggle search method
 */
function toggleSearchMethod() {
    searchMethod = searchMethod === 'vector' ? 'keyword' : 'vector';
    updateSearchMethodUI();
    showAlert(
        `검색 방법이 ${searchMethod === 'vector' ? '벡터 검색' : '키워드 검색'}으로 변경되었습니다.`,
        'info',
        3000
    );
}

/**
 * Update search method UI
 */
function updateSearchMethodUI() {
    const toggleButton = document.getElementById('method-toggle');
    if (toggleButton) {
        toggleButton.textContent = searchMethod === 'vector' ? '키워드 검색으로 변경' : '벡터 검색으로 변경';
        toggleButton.className = `btn btn-sm ${searchMethod === 'vector' ? 'btn-secondary' : 'btn-primary'}`;
    }
}

// Event Listeners
if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const question = questionInput.value.trim();
        if (!question) {
            showAlert('질문을 입력해주세요.', 'warning');
            return;
        }

        await sendQuestion(question);
        questionInput.value = '';
    });
}

// Initialize page-specific functionality
document.addEventListener('DOMContentLoaded', () => {
    const isAdminPage = window.location.pathname.includes('/admin');

    if (isAdminPage) {
        initAdminPage();
    } else {
        initMainPage();
    }
});

/**
 * Initialize main page
 */
function initMainPage() {
    // 검색 방법 토글 버튼 추가
    const chatInput = document.querySelector('.chat-input');
    if (chatInput) {
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'method-toggle-container';
        toggleContainer.style.marginTop = '10px';
        toggleContainer.style.textAlign = 'center';

        const toggleButton = document.createElement('button');
        toggleButton.id = 'method-toggle';
        toggleButton.type = 'button';
        toggleButton.onclick = toggleSearchMethod;

        toggleContainer.appendChild(toggleButton);
        chatInput.appendChild(toggleContainer);

        updateSearchMethodUI();
    }

    // 초기 메시지에 검색 방법 안내 추가
    setTimeout(() => {
        addMessage(
            "벡터 검색과 키워드 검색 두 가지 방법을 사용할 수 있습니다. 버튼을 클릭하여 검색 방법을 변경할 수 있습니다.",
            'bot'
        );
    }, 500);
}

/**
 * Initialize admin page functionality
 */
function initAdminPage() {
    // Admin page elements
    const dataTableBody = document.getElementById('data-table-body');
    const addEntryForm = document.getElementById('add-entry-form');
    const questionField = document.getElementById('question-field');
    const responseField = document.getElementById('response-field');
    const modal = document.getElementById('entry-modal');
    const editForm = document.getElementById('edit-entry-form');

    // Pagination
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const currentPageSpan = document.getElementById('current-page');

    let currentPage = 1;
    const entriesPerPage = 10;

    /**
     * Load entries from API
     */
    async function loadEntries() {
        try {
            const offset = (currentPage - 1) * entriesPerPage;

            const response = await fetch(
                `${API_ENDPOINTS.DATA}?limit=${entriesPerPage}&offset=${offset}&sort_by=created_at&sort_order=DESC`
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.state) {
                dataTableBody.innerHTML = '';

                if (data.entries && data.entries.length > 0) {
                    data.entries.forEach(entry => {
                        addEntryToTable(entry);
                    });
                    updatePagination(data.total);
                } else {
                    dataTableBody.innerHTML = `
                        <tr>
                            <td colspan="5" class="text-center">등록된 항목이 없습니다</td>
                        </tr>
                    `;
                }
            } else {
                showAlert(`데이터 로딩 실패: ${data.error || '알 수 없는 오류'}`, 'danger');
            }
        } catch (error) {
            console.error('Error loading entries:', error);
            showAlert(`로딩 오류: ${error.message}`, 'danger');
        }
    }

    /**
     * Add entry to table
     */
    function addEntryToTable(entry) {
        const tr = document.createElement('tr');

        // 날짜 포맷팅
        const createdDate = new Date(entry.created_at * 1000).toLocaleString('ko-KR');
        const lastAccessedDate = new Date(entry.last_accessed * 1000).toLocaleString('ko-KR');

        // 질문/답변 텍스트 길이 제한
        const shortQuestion = entry.question.length > 50
            ? entry.question.substring(0, 50) + '...'
            : entry.question;
        const shortResponse = entry.response.length > 100
            ? entry.response.substring(0, 100) + '...'
            : entry.response;

        tr.innerHTML = `
            <td title="${entry.question}">${shortQuestion}</td>
            <td title="${entry.response}">${shortResponse}</td>
            <td>${createdDate}</td>
            <td>${entry.hits}</td>
            <td>
                <button class="btn btn-sm btn-primary edit-btn" data-id="${entry.id}" title="수정">
                    ✏️ 수정
                </button>
                <button class="btn btn-sm btn-danger delete-btn" data-id="${entry.id}" title="삭제">
                    🗑️ 삭제
                </button>
            </td>
        `;

        dataTableBody.appendChild(tr);

        // Event listeners
        tr.querySelector('.edit-btn').addEventListener('click', () => {
            openEditModal(entry);
        });

        tr.querySelector('.delete-btn').addEventListener('click', () => {
            deleteEntry(entry.id);
        });
    }

    /**
     * Update pagination controls
     */
    function updatePagination(totalEntries) {
        const totalPages = Math.ceil(totalEntries / entriesPerPage);

        if (currentPageSpan) {
            currentPageSpan.textContent = `페이지 ${currentPage} / ${totalPages} (총 ${totalEntries}개)`;
        }

        if (prevPageButton) {
            prevPageButton.disabled = currentPage <= 1;
        }
        if (nextPageButton) {
            nextPageButton.disabled = currentPage >= totalPages;
        }
    }

    /**
     * Open edit modal
     */
    function openEditModal(entry) {
        const editQuestionField = document.getElementById('edit-question-field');
        const editResponseField = document.getElementById('edit-response-field');
        const editEntryId = document.getElementById('edit-entry-id');

        if (editQuestionField && editResponseField && editEntryId) {
            editQuestionField.value = entry.question;
            editResponseField.value = entry.response;
            editEntryId.value = entry.id;

            if (modal) {
                modal.style.display = 'block';
            }
        }
    }

    /**
     * Delete entry
     */
    async function deleteEntry(id) {
        if (!confirm('정말로 이 항목을 삭제하시겠습니까?')) {
            return;
        }

        try {
            const response = await fetch(API_ENDPOINTS.DATA, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ id })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.state) {
                showAlert('항목이 성공적으로 삭제되었습니다!', 'success');
                await loadEntries(); // 목록 새로고침
            } else {
                showAlert(`삭제 실패: ${data.error || '알 수 없는 오류'}`, 'danger');
            }
        } catch (error) {
            console.error('Error deleting entry:', error);
            showAlert(`삭제 오류: ${error.message}`, 'danger');
        }
    }

    // Form submit handlers
    if (addEntryForm) {
        addEntryForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const question = questionField.value.trim();
            const method = document.getElementById('method-field')?.value || 'vector';
            const response = responseField.value.trim();

            if (!question || !response) {
                showAlert('질문과 답변을 모두 입력해주세요.', 'warning');
                return;
            }

            const resultId = await storeQuestionResponse(question, method, response);

            if (resultId) {
                questionField.value = '';
                document.getElementById('method-field').value = 'vector';
                responseField.value = '';
                await loadEntries(); // 목록 새로고침
            }
        });
    }

    if (editForm) {
        editForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const editQuestionField = document.getElementById('edit-question-field');
            const editResponseField = document.getElementById('edit-response-field');
            const editEntryId = document.getElementById('edit-entry-id');

            const id = editEntryId.value;
            const question = editQuestionField.value.trim();
            const response = editResponseField.value.trim();

            if (!id || !question || !response) {
                showAlert('모든 필드를 입력해주세요.', 'warning');
                return;
            }

            try {
                // 수정은 삭제 후 새로 등록하는 방식으로 구현
                // (Redis Search의 특성상 직접 수정이 복잡함)
                showAlert('수정 기능은 현재 구현 중입니다.', 'info');

                if (modal) {
                    modal.style.display = 'none';
                }
            } catch (error) {
                console.error('Error updating entry:', error);
                showAlert(`수정 오류: ${error.message}`, 'danger');
            }
        });
    }

    // Modal close handlers
    const modalCloseButtons = document.querySelectorAll('.modal-close');
    modalCloseButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (modal) {
                modal.style.display = 'none';
            }
        });
    });

    // Close modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Pagination handlers
    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                loadEntries();
            }
        });
    }

    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            currentPage++;
            loadEntries();
        });
    }

    // 통계 정보 로드
    async function loadStats() {
        try {
            const response = await fetch(API_ENDPOINTS.STATS);
            const data = await response.json();

            if (data.state) {
                console.log('Cache stats:', data.stats);
                // 통계 정보를 UI에 표시할 수 있습니다
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    // 초기 로드
    loadEntries();
    loadStats();
}

// CSS 스타일 추가 (동적으로)
const style = document.createElement('style');
style.textContent = `
    .cached-response {
        border-left: 4px solid #4caf50;
        background-color: #f8fff8;
    }
    
    .search-method-badge {
        background-color: #007bff;
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.7em;
        margin-left: 8px;
    }
    
    .method-toggle-container {
        margin-top: 10px;
        text-align: center;
    }
    
    .alert {
        position: relative;
        padding: 12px 40px 12px 16px;
        margin-bottom: 16px;
        border-radius: 4px;
        border: 1px solid transparent;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .alert-close {
        position: absolute;
        top: 50%;
        right: 12px;
        transform: translateY(-50%);
        background: none;
        border: none;
        font-size: 18px;
        cursor: pointer;
        padding: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: color 0.2s;
    }
    
    .alert-close:hover {
        color: #000;
    }
    
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        border-color: #ffeeba;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    
    .message-info {
        font-size: 0.85em;
        color: #666;
        margin-bottom: 5px;
        font-weight: 500;
    }
    
    .message-content {
        line-height: 1.5;
        word-break: break-word;
    }
    
    .btn-sm {
        padding: 4px 8px;
        font-size: 12px;
        margin: 0 2px;
    }
    
    .text-center {
        text-align: center;
    }
    
    .input-container {
        display: flex;
        gap: 10px;
        align-items: flex-end;
    }
    
    .input-container .form-control {
        flex: 1;
    }
    
    .pagination-container {
        margin-top: 20px;
        text-align: center;
    }
    
    .pagination-controls {
        display: inline-flex;
        align-items: center;
        gap: 15px;
    }
`;
document.head.appendChild(style);